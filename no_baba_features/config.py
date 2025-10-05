"""
Configuration file for BABA Premium/Discount Prediction System
"""
import os
from datetime import datetime

# ============================================================================
# DATA PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# CSV Data Files
MINUTE_DATA_FILE = os.path.join(BASE_DIR, 'baba_all_minute_data.csv')
OPTION_DATA_FILE = os.path.join(BASE_DIR, 'baba_all_option_trades.csv')

# Output Files
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, 'processed_data.parquet')
FEATURES_FILE = os.path.join(DATA_DIR, 'features.parquet')
TRAIN_DATA_FILE = os.path.join(DATA_DIR, 'train_data.parquet')
VAL_DATA_FILE = os.path.join(DATA_DIR, 'val_data.parquet')
TEST_DATA_FILE = os.path.join(DATA_DIR, 'test_data.parquet')

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# BLOOMBERG DATA CONFIGURATION
# ============================================================================
# Tickers
BABA_US_TICKER = 'BABA US Equity'
BABA_HK_TICKER = '9988 HK Equity'
USDHKD_TICKER = 'USDHKD Curncy'
VIX_TICKER = 'VIX Index'
US_10Y_TICKER = 'USGG10YR Index'
US_3M_TICKER = 'USGG3M Index'
PDD_TICKER = 'PDD US Equity'

# Bloomberg Fields
PRICE_FIELDS = ['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST', 'PX_VOLUME']
OPTION_FIELDS = ['IVOL_MID']  # Implied Volatility

# ADR Conversion Ratio (BABA ADR to HK shares)
ADR_CONVERSION_RATIO = 8  # 1 BABA ADR = 8 9988.HK shares

# ============================================================================
# TIME CONFIGURATION
# ============================================================================
# Timezones
US_TIMEZONE = 'America/New_York'
HK_TIMEZONE = 'Asia/Hong_Kong'

# Trading Hours (for intraday analysis)
US_MARKET_OPEN = '09:30'
US_MARKET_CLOSE = '16:00'
US_FIRST_30MIN_END = '10:00'
US_LAST_30MIN_START = '15:30'

# Data Date Range
START_DATE = '2014-09-19'  # BABA IPO date
END_DATE = datetime.now().strftime('%Y-%m-%d')

# ============================================================================
# FEATURE ENGINEERING PARAMETERS
# ============================================================================
# Lookback periods
PREM_DISCOUNT_LOOKBACK = 5  # days
PRICE_CHANGE_LOOKBACK = 5  # days
VOLUME_AVG_WINDOW = 30  # days
REALIZED_VOL_WINDOW = 5  # days
RSI_PERIOD = 5  # days

# Feature groups - NO BABA FEATURES VERSION
FEATURE_GROUPS = {
    'historical_premium': ['prem_discount_lag_{}'.format(i) for i in range(1, PREM_DISCOUNT_LOOKBACK + 1)],
    'hk_price_momentum': ['hk_pct_change_t'] + ['hk_pct_change_lag_{}'.format(i) for i in range(1, PRICE_CHANGE_LOOKBACK + 1)],
    'volume_features': ['hk_volume_ratio'],  # REMOVED: baba_volume_ratio
    # REMOVED: 'intraday_features': ['baba_last_30min_pct_change', 'baba_first_30min_volume']
    # REMOVED: 'volatility_features': ['baba_realized_vol_5d']
    # REMOVED: 'option_features': ['baba_option_volume']
    'technical_indicators': ['hk_rsi_{}'.format(RSI_PERIOD)],
    'macro_features': ['us_10y_yield', 'us_3m_yield', 'vix_level', 'usdhkd_pct_change'],
    'peer_features': ['pdd_pct_change']
}

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Train/Validation/Test Split (time-series aware)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_STATE = 42

# Classification threshold (for binary classification)
CLASSIFICATION_THRESHOLD = 0.0  # Premium if > 0, Discount if < 0

# ============================================================================
# XGBOOST HYPERPARAMETERS
# ============================================================================
XGBOOST_PARAMS_CLASSIFICATION = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

XGBOOST_PARAMS_REGRESSION = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

# ============================================================================
# LIGHTGBM HYPERPARAMETERS
# ============================================================================
LIGHTGBM_PARAMS_CLASSIFICATION = {
    'num_leaves': [31, 50, 70],
    'max_depth': [5, 7, 9, -1],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'min_child_samples': [20, 30, 50],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

LIGHTGBM_PARAMS_REGRESSION = {
    'num_leaves': [31, 50, 70],
    'max_depth': [5, 7, 9, -1],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'min_child_samples': [20, 30, 50],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

# ============================================================================
# RANDOM FOREST HYPERPARAMETERS
# ============================================================================
RF_PARAMS_CLASSIFICATION = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

RF_PARAMS_REGRESSION = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# ============================================================================
# NEURAL NETWORK HYPERPARAMETERS
# ============================================================================
NN_PARAMS = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [500, 1000]
}

# ============================================================================
# EVALUATION METRICS
# ============================================================================
CLASSIFICATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'roc_auc'
]

REGRESSION_METRICS = [
    'rmse',
    'mae',
    'r2',
    'mape'
]

# ============================================================================
# OPTUNA CONFIGURATION
# ============================================================================
OPTUNA_N_TRIALS = 100
OPTUNA_TIMEOUT = 3600  # 1 hour in seconds
OPTUNA_N_JOBS = -1  # Use all available cores

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = 'INFO'
LOG_FILE = os.path.join(RESULTS_DIR, 'training.log')

