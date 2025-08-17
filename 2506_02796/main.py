#!/usr/bin/env python3
"""
LSTM-BEKK Trading System - Main Execution Script

This script demonstrates the complete LSTM-BEKK trading system implementation
based on the research paper "Deep Learning Enhanced Multivariate GARCH".

Usage:
    python main.py --mode backtest --universe sp500_sample
    python main.py --mode dashboard --results results.pkl
    python main.py --mode live --universe sp500_sample
"""

import argparse
import logging
import yaml
import pickle
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from lstm_bekk_trading import (
    DataManager, LSTMBEKKModel, TradingStrategies, 
    RiskManager, BacktestEngine, Visualizer
)
from lstm_bekk_trading.visualization import Dashboard


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('lstm_bekk_trading.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file {config_path} not found")
        return {}


def run_backtest(config: dict, universe: str = "sp500_sample", 
                start_date: str = None, end_date: str = None) -> dict:
    """
    Run comprehensive backtest of LSTM-BEKK trading system.
    
    Args:
        config: Configuration dictionary
        universe: Stock universe to test
        start_date: Backtest start date
        end_date: Backtest end date
        
    Returns:
        Backtest results
    """
    logging.info("=" * 60)
    logging.info("LSTM-BEKK TRADING SYSTEM - BACKTEST MODE")
    logging.info("=" * 60)
    
    # Initialize components
    data_manager = DataManager(config=config)
    backtest_engine = BacktestEngine(config)
    
    # Load and prepare data
    logging.info(f"Loading data for universe: {universe}")
    data_manager.load_data(universe, start_date, end_date)
    
    # Display data statistics
    stats = data_manager.get_data_statistics()
    logging.info(f"Data loaded: {stats['shape'][0]} periods, {stats['shape'][1]} assets")
    logging.info(f"Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
    
    # Validate data quality
    validation = data_manager.validate_data_quality()
    if not all(validation.values()):
        logging.warning("Data quality issues detected:")
        for check, passed in validation.items():
            if not passed:
                logging.warning(f"  - {check}: FAILED")
    
    # Run backtest
    logging.info("Starting comprehensive backtest...")
    results = backtest_engine.run_backtest(data_manager, start_date, end_date, universe)
    
    # Display results summary
    logging.info("\n" + "=" * 50)
    logging.info("BACKTEST RESULTS SUMMARY")
    logging.info("=" * 50)
    
    if 'performance_comparison' in results:
        comparison = results['performance_comparison']
        logging.info("\nStrategy Performance Ranking (by Sharpe Ratio):")
        
        sorted_strategies = comparison.sort_values('Sharpe Ratio', ascending=False)
        for i, (strategy, metrics) in enumerate(sorted_strategies.iterrows(), 1):
            logging.info(f"{i:2d}. {strategy:20s} | "
                        f"Sharpe: {metrics['Sharpe Ratio']:6.3f} | "
                        f"Return: {metrics['Annualized Return']:7.2%} | "
                        f"Vol: {metrics['Annualized Volatility']:6.2%} | "
                        f"MDD: {metrics['Max Drawdown']:7.2%}")
    
    # LSTM-BEKK specific results
    if 'lstm_bekk' in results['strategy_results']:
        lstm_results = results['strategy_results']['lstm_bekk']
        logging.info(f"\nLSTM-BEKK Model Details:")
        logging.info(f"  - Training epochs: {len(lstm_results.get('training_history', []))}")
        logging.info(f"  - Model refits: {len(lstm_results.get('refit_dates', []))}")
        
        if 'performance_comparison' in results and 'LSTM-BEKK' in results['performance_comparison'].index:
            lstm_metrics = results['performance_comparison'].loc['LSTM-BEKK']
            logging.info(f"  - Final Sharpe ratio: {lstm_metrics['Sharpe Ratio']:.3f}")
            logging.info(f"  - Sharpe rank: {lstm_metrics['Sharpe Rank']:.0f}")
    
    return results


def run_dashboard(results_path: str = None, backtest_results: dict = None):
    """
    Launch interactive dashboard.
    
    Args:
        results_path: Path to saved backtest results
        backtest_results: Backtest results dictionary
    """
    logging.info("=" * 60)
    logging.info("LSTM-BEKK TRADING SYSTEM - DASHBOARD MODE")
    logging.info("=" * 60)
    
    # Load results if path provided
    if results_path and Path(results_path).exists():
        logging.info(f"Loading results from {results_path}")
        with open(results_path, 'rb') as f:
            backtest_results = pickle.load(f)
    
    if backtest_results is None:
        logging.error("No backtest results available for dashboard")
        return
    
    # Launch dashboard
    dashboard = Dashboard(backtest_results)
    logging.info("Launching interactive dashboard...")
    dashboard.run(debug=False)


def run_live_trading(config: dict, universe: str = "sp500_sample"):
    """
    Run live trading simulation.
    
    Args:
        config: Configuration dictionary
        universe: Stock universe to trade
    """
    logging.info("=" * 60)
    logging.info("LSTM-BEKK TRADING SYSTEM - LIVE TRADING MODE")
    logging.info("=" * 60)
    
    # Initialize components
    data_manager = DataManager(config=config)
    risk_manager = RiskManager(config)
    trading_strategies = TradingStrategies(config)
    
    # Load recent data
    logging.info(f"Loading recent data for universe: {universe}")
    data_manager.load_data(universe)
    
    # Initialize model
    n_assets = len(data_manager.get_returns().columns)
    model = LSTMBEKKModel(n_assets, config)
    
    # Train model on available data
    train_data = data_manager.get_train_data()
    val_data = data_manager.get_validation_data()
    
    logging.info("Training LSTM-BEKK model for live trading...")
    model.fit(train_data, val_data, verbose=True)
    
    # Simulate live trading loop
    test_data = data_manager.get_test_data()
    current_portfolio_value = config.get('trading', {}).get('portfolio', {}).get('initial_capital', 1000000)
    
    logging.info(f"Starting live trading simulation with ${current_portfolio_value:,.0f}")
    
    for i, (date, returns) in enumerate(test_data.iterrows()):
        if i % 5 == 0:  # Rebalance every 5 days
            # Get historical data
            historical_data = data_manager.get_returns().loc[:date].tail(252)
            
            # Generate covariance forecast
            cov_forecast = model.predict_covariance(historical_data, steps=1)[0]
            
            # Generate trading signals
            model_outputs = {
                'covariance_forecast': cov_forecast,
                'volatility_forecast': np.sqrt(np.diag(cov_forecast)),
                'correlation_forecast': cov_forecast / np.outer(np.sqrt(np.diag(cov_forecast)), 
                                                               np.sqrt(np.diag(cov_forecast)))
            }
            
            signals = trading_strategies.generate_trading_signals(
                model_outputs, historical_data, date
            )
            
            # Risk assessment
            if signals['final_weights'] is not None:
                risk_assessment = risk_manager.assess_portfolio_risk(
                    signals['final_weights'], cov_forecast, historical_data, date
                )
                
                # Log risk alerts
                if risk_assessment['risk_alerts']:
                    for alert in risk_assessment['risk_alerts']:
                        logging.warning(f"Risk Alert: {alert['message']}")
                
                # Execute rebalancing if required
                if signals['rebalance_required']:
                    current_prices = historical_data.iloc[-1] + 100  # Simulate prices
                    
                    execution_result = trading_strategies.execute_rebalancing(
                        signals['final_weights'], current_prices, date, current_portfolio_value
                    )
                    
                    if execution_result['success']:
                        logging.info(f"Rebalanced portfolio on {date}: "
                                   f"{len(execution_result['trades'])} trades, "
                                   f"${execution_result['total_trade_value']:,.0f} total value")
        
        # Update portfolio value (simplified)
        if len(trading_strategies.position_history) > 0:
            weights = trading_strategies.position_history[-1]['weights']
            portfolio_return = np.sum(weights * returns)
            current_portfolio_value *= (1 + portfolio_return)
    
    # Final performance summary
    performance = trading_strategies.get_strategy_performance()
    logging.info("\n" + "=" * 50)
    logging.info("LIVE TRADING SIMULATION RESULTS")
    logging.info("=" * 50)
    logging.info(f"Final portfolio value: ${current_portfolio_value:,.0f}")
    logging.info(f"Total return: {performance.get('total_return', 0):.2%}")
    logging.info(f"Sharpe ratio: {performance.get('sharpe_ratio', 0):.3f}")
    logging.info(f"Max drawdown: {performance.get('max_drawdown', 0):.2%}")
    logging.info(f"Number of rebalances: {performance.get('num_rebalances', 0)}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="LSTM-BEKK Trading System")
    parser.add_argument('--mode', choices=['backtest', 'dashboard', 'live'], 
                       default='backtest', help='Execution mode')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--universe', default='sp500_sample', help='Stock universe to use')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--results', help='Path to saved backtest results (for dashboard mode)')
    parser.add_argument('--save-results', help='Path to save backtest results')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    try:
        if args.mode == 'backtest':
            # Run backtest
            results = run_backtest(config, args.universe, args.start_date, args.end_date)
            
            # Save results if requested
            if args.save_results:
                with open(args.save_results, 'wb') as f:
                    pickle.dump(results, f)
                logging.info(f"Results saved to {args.save_results}")
            
            # Optionally launch dashboard
            response = input("\nLaunch interactive dashboard? (y/n): ")
            if response.lower() == 'y':
                run_dashboard(backtest_results=results)
        
        elif args.mode == 'dashboard':
            run_dashboard(args.results)
        
        elif args.mode == 'live':
            run_live_trading(config, args.universe)
    
    except KeyboardInterrupt:
        logging.info("Execution interrupted by user")
    except Exception as e:
        logging.error(f"Execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
