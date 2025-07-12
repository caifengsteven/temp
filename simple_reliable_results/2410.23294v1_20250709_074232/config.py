"""
Configuration file for Bloomberg Forex Trading Strategy
Modify these settings to customize the trading strategy behavior
"""

# Data Configuration
DATA_CONFIG = {
    'use_bloomberg': True,  # Set to False to use synthetic data
    'currency_pair': 'EURUSD Curncy',  # Bloomberg ticker
    'start_date': '2018-01-01',
    'end_date': '2022-12-31',
    'fallback_to_synthetic': True,  # Use synthetic data if Bloomberg fails
}

# Training Configuration
TRAINING_CONFIG = {
    'episodes': 50,  # Number of training episodes (increase for better performance)
    'batch_size': 64,  # Batch size for model training
    'evaluate_every': 5,  # Evaluate model every N episodes
    'persistence_values': [1, 5, 10],  # Action persistence values to test
}

# Model Configuration
MODEL_CONFIG = {
    'learning_rate': 0.001,
    'gamma': 1.0,  # Discount factor
    
    # XGBoost parameters for value function
    'xgb_params': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'min_child_weight': 1,
        'gamma': 0.1,
        'reg_lambda': 1.0
    },
    
    # Ridge regression parameters for advantage function
    'ridge_params': {
        'alpha': 0.1
    },
    
    # Neural network parameters for policy
    'nn_params_discrete': {
        'hidden_layer_sizes': (64,),
        'activation': 'relu',
        'learning_rate_init': 0.001,
        'max_iter': 1000
    },
    
    'nn_params_continuous': {
        'hidden_layer_sizes': (64, 32),
        'activation': 'relu',
        'learning_rate_init': 0.001,
        'max_iter': 1000
    }
}

# Risk Management Configuration
RISK_CONFIG = {
    'test_risk_averse': True,  # Whether to test risk-averse variants
    'mean_volatility_lambda': 0.001,  # λ parameter for Mean-Volatility
    'rcvar_rho': 0.5,  # ρ parameter for RCVaR
}

# Transaction Cost Configuration
TRANSACTION_CONFIG = {
    'test_variable_fees': True,  # Whether to test variable fee structure
    'base_spread_pips': 2.0,  # Base spread in pips
    'fixed_fee_multiplier': 0.5,  # Fixed fee multiplier
    
    # Variable fee structure (order_size -> multiplier)
    'variable_fee_structure': {
        0.2: 0.5,   # Small orders: 0.5x multiplier
        0.5: 1.0,   # Medium orders: 1.0x multiplier  
        0.8: 1.5,   # Large orders: 1.5x multiplier
        1.0: 2.0    # Very large orders: 2.0x multiplier
    }
}

# Multi-Currency Testing Configuration
MULTI_CURRENCY_CONFIG = {
    'currency_pairs': [
        'EURUSD Curncy',  # EUR/USD
        'GBPUSD Curncy',  # GBP/USD  
        'USDJPY Curncy',  # USD/JPY
        'AUDUSD Curncy',  # AUD/USD
        'USDCAD Curncy'   # USD/CAD
    ],
    'quick_test_episodes': 20,  # Reduced episodes for multi-currency testing
    'min_data_points': 1000,    # Minimum data points required per currency
    'min_train_points': 500,    # Minimum training data points
    'min_test_points': 100,     # Minimum test data points
}

# Output Configuration
OUTPUT_CONFIG = {
    'save_models': True,        # Whether to save trained models
    'generate_reports': True,   # Whether to generate analysis reports
    'show_plots': True,         # Whether to display matplotlib plots
    'verbose_logging': True,    # Detailed console output
}

# Bloomberg Specific Configuration
BLOOMBERG_CONFIG = {
    'intraday_bar_size': '1T',  # 1-minute bars
    'trading_hours_start': 8,   # Trading day start hour
    'trading_hours_end': 18,    # Trading day end hour
    'volume_log_scale': True,   # Use log scaling for volume features
    'include_vwap': True,       # Include VWAP deviation features
    'include_hl_range': True,   # Include high-low range features
}

# Advanced Configuration
ADVANCED_CONFIG = {
    'random_seed': 42,          # Random seed for reproducibility
    'parallel_processing': False,  # Enable parallel processing (experimental)
    'memory_optimization': True,   # Enable memory optimization techniques
    'early_stopping': True,       # Enable early stopping in training
    'cross_validation': False,    # Enable cross-validation (slower)
}

def get_config():
    """Return complete configuration dictionary"""
    return {
        'data': DATA_CONFIG,
        'training': TRAINING_CONFIG,
        'model': MODEL_CONFIG,
        'risk': RISK_CONFIG,
        'transaction': TRANSACTION_CONFIG,
        'multi_currency': MULTI_CURRENCY_CONFIG,
        'output': OUTPUT_CONFIG,
        'bloomberg': BLOOMBERG_CONFIG,
        'advanced': ADVANCED_CONFIG
    }

def print_config():
    """Print current configuration"""
    config = get_config()
    print("Current Configuration:")
    print("=" * 50)
    
    for section, settings in config.items():
        print(f"\n{section.upper()}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    print_config()
