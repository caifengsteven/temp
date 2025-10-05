"""
Main Orchestration Script - Run the complete ML pipeline
"""
import pandas as pd
import numpy as np
import logging
import argparse
import os
from datetime import datetime

import config
from data_collector import DataCollector
from csv_processor import CSVProcessor
from feature_engineering import FeatureEngineer
from data_preprocessing import DataPreprocessor
from model_trainer import ModelTrainer
from evaluation import ModelEvaluator
from predict import Predictor, EnsemblePredictor

# Setup logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def collect_data(start_date: str = None, end_date: str = None):
    """
    Step 1: Collect data from Bloomberg and CSV files
    
    Args:
        start_date: Start date for data collection
        end_date: End date for data collection
    """
    logger.info("=" * 80)
    logger.info("STEP 1: DATA COLLECTION")
    logger.info("=" * 80)
    
    # Collect Bloomberg data
    collector = DataCollector(start_date=start_date, end_date=end_date)
    data = collector.collect_all_data()
    collector.save_data(data)
    
    # SKIP CSV processing - NO BABA features version doesn't need intraday data
    logger.info("SKIPPING CSV data processing (NO BABA features version - no intraday features needed)")
    # Create empty intraday features dataframe (not used in NO BABA version)
    intraday_features = pd.DataFrame()
    logger.info("No CSV intraday features needed for NO BABA features version")
    
    logger.info("Data collection completed")


def engineer_features():
    """
    Step 2: Engineer features from raw data
    """
    logger.info("=" * 80)
    logger.info("STEP 2: FEATURE ENGINEERING")
    logger.info("=" * 80)
    
    # Load data
    data = {}
    data_dir = config.DATA_DIR
    
    for name in ['baba_us', 'baba_hk', 'usdhkd', 'vix', 'treasury', 'pdd', 'common_days', 'implied_vol']:
        file_path = f"{data_dir}/{name}.parquet"
        if os.path.exists(file_path):
            data[name] = pd.read_parquet(file_path)
            logger.info(f"Loaded {name} data: {len(data[name])} records")
    
    # Load intraday features
    intraday_file = f"{data_dir}/intraday_features.parquet"
    if os.path.exists(intraday_file):
        data['intraday_features'] = pd.read_parquet(intraday_file)
        logger.info(f"Loaded intraday features: {len(data['intraday_features'])} records")
    
    # Create features
    engineer = FeatureEngineer()
    features_df = engineer.create_all_features(data)
    
    # Save features
    features_df.to_parquet(config.FEATURES_FILE, index=False)
    logger.info(f"Saved features to {config.FEATURES_FILE}")
    logger.info(f"Feature matrix shape: {features_df.shape}")
    
    return features_df


def preprocess_data(features_df: pd.DataFrame = None):
    """
    Step 3: Preprocess data (clean, scale, split)
    
    Args:
        features_df: Features DataFrame (optional, will load from file if not provided)
    """
    logger.info("=" * 80)
    logger.info("STEP 3: DATA PREPROCESSING")
    logger.info("=" * 80)
    
    # Load features if not provided
    if features_df is None:
        if os.path.exists(config.FEATURES_FILE):
            features_df = pd.read_parquet(config.FEATURES_FILE)
        else:
            raise FileNotFoundError(f"Features file not found: {config.FEATURES_FILE}")
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    train_df, val_df, test_df = preprocessor.prepare_data_for_training(features_df)
    
    # Save preprocessed data
    train_df.to_parquet(config.TRAIN_DATA_FILE, index=False)
    val_df.to_parquet(config.VAL_DATA_FILE, index=False)
    test_df.to_parquet(config.TEST_DATA_FILE, index=False)
    
    # Save preprocessor
    preprocessor.save_preprocessor()
    
    logger.info("Data preprocessing completed")
    
    return train_df, val_df, test_df


def train_models(use_optuna: bool = True):
    """
    Step 4: Train all models
    
    Args:
        use_optuna: Use Optuna for hyperparameter tuning
    """
    logger.info("=" * 80)
    logger.info("STEP 4: MODEL TRAINING")
    logger.info("=" * 80)
    
    # Load preprocessed data
    train_df = pd.read_parquet(config.TRAIN_DATA_FILE)
    val_df = pd.read_parquet(config.VAL_DATA_FILE)
    
    # Load feature columns
    import joblib
    feature_columns = joblib.load(f"{config.MODEL_DIR}/feature_columns.pkl")
    
    # Train classification models
    logger.info("-" * 80)
    logger.info("TRAINING CLASSIFICATION MODELS")
    logger.info("-" * 80)
    
    clf_trainer = ModelTrainer(task_type='classification')
    clf_models = clf_trainer.train_all_models(train_df, val_df, feature_columns, use_optuna=use_optuna)
    clf_trainer.save_models()
    
    # Train regression models
    logger.info("-" * 80)
    logger.info("TRAINING REGRESSION MODELS")
    logger.info("-" * 80)
    
    reg_trainer = ModelTrainer(task_type='regression')
    reg_models = reg_trainer.train_all_models(train_df, val_df, feature_columns, use_optuna=use_optuna)
    reg_trainer.save_models()
    
    logger.info("Model training completed")


def evaluate_models():
    """
    Step 5: Evaluate and compare models
    """
    logger.info("=" * 80)
    logger.info("STEP 5: MODEL EVALUATION")
    logger.info("=" * 80)
    
    # Load data
    train_df = pd.read_parquet(config.TRAIN_DATA_FILE)
    val_df = pd.read_parquet(config.VAL_DATA_FILE)
    test_df = pd.read_parquet(config.TEST_DATA_FILE)
    
    # Load feature columns
    import joblib
    feature_columns = joblib.load(f"{config.MODEL_DIR}/feature_columns.pkl")
    
    # Prepare data
    trainer = ModelTrainer(task_type='classification')
    X_train, y_train = trainer.prepare_data(train_df, feature_columns)
    X_val, y_val = trainer.prepare_data(val_df, feature_columns)
    X_test, y_test = trainer.prepare_data(test_df, feature_columns)
    
    # Evaluate classification models
    logger.info("-" * 80)
    logger.info("EVALUATING CLASSIFICATION MODELS")
    logger.info("-" * 80)
    
    clf_models = {}
    for model_name in ['xgboost', 'lightgbm', 'random_forest', 'logistic_regression', 'neural_network']:
        model_path = f"{config.MODEL_DIR}/{model_name}_classification.pkl"
        if os.path.exists(model_path):
            clf_models[model_name] = joblib.load(model_path)
    
    clf_evaluator = ModelEvaluator(task_type='classification')
    clf_results = clf_evaluator.evaluate_all_models(clf_models, X_train, y_train, X_val, y_val, X_test, y_test)
    
    print("\nClassification Results:")
    print(clf_results.to_string())
    
    # Select best classification model
    best_clf_name, best_clf_metrics = clf_evaluator.select_best_model(clf_results)
    
    # Plot comparison
    clf_evaluator.plot_model_comparison(clf_results)
    
    # Plot feature importance for best model
    clf_evaluator.plot_feature_importance(clf_models[best_clf_name], feature_columns, best_clf_name)
    
    # Save results
    clf_evaluator.save_results(clf_results)
    
    # Evaluate regression models
    logger.info("-" * 80)
    logger.info("EVALUATING REGRESSION MODELS")
    logger.info("-" * 80)
    
    reg_trainer = ModelTrainer(task_type='regression')
    X_train_reg, y_train_reg = reg_trainer.prepare_data(train_df, feature_columns)
    X_val_reg, y_val_reg = reg_trainer.prepare_data(val_df, feature_columns)
    X_test_reg, y_test_reg = reg_trainer.prepare_data(test_df, feature_columns)
    
    reg_models = {}
    for model_name in ['xgboost', 'lightgbm', 'random_forest', 'ridge', 'neural_network']:
        model_path = f"{config.MODEL_DIR}/{model_name}_regression.pkl"
        if os.path.exists(model_path):
            reg_models[model_name] = joblib.load(model_path)
    
    reg_evaluator = ModelEvaluator(task_type='regression')
    reg_results = reg_evaluator.evaluate_all_models(reg_models, X_train_reg, y_train_reg, X_val_reg, y_val_reg, X_test_reg, y_test_reg)
    
    print("\nRegression Results:")
    print(reg_results.to_string())
    
    # Select best regression model
    best_reg_name, best_reg_metrics = reg_evaluator.select_best_model(reg_results)
    
    # Plot comparison
    reg_evaluator.plot_model_comparison(reg_results)
    
    # Plot feature importance for best model
    reg_evaluator.plot_feature_importance(reg_models[best_reg_name], feature_columns, best_reg_name)
    
    # Save results
    reg_evaluator.save_results(reg_results)
    
    logger.info("Model evaluation completed")
    
    return best_clf_name, best_reg_name


def make_predictions(model_name: str = 'xgboost', task_type: str = 'classification'):
    """
    Step 6: Make predictions on test data
    
    Args:
        model_name: Name of the model to use
        task_type: 'classification' or 'regression'
    """
    logger.info("=" * 80)
    logger.info(f"STEP 6: MAKING PREDICTIONS ({task_type.upper()})")
    logger.info("=" * 80)
    
    # Load test data
    test_df = pd.read_parquet(config.TEST_DATA_FILE)
    
    # Load model
    model_path = f"{config.MODEL_DIR}/{model_name}_{task_type}.pkl"
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return
    
    # Make predictions
    predictor = Predictor(model_path, task_type=task_type)
    predictions = predictor.backtest_predictions(test_df)
    
    # Save predictions
    predictor.save_predictions(predictions)
    
    print(f"\n{task_type.capitalize()} Predictions (first 10):")
    print(predictions.head(10).to_string())
    
    logger.info("Predictions completed")
    
    return predictions


def run_full_pipeline(start_date: str = None, end_date: str = None, use_optuna: bool = True):
    """
    Run the complete ML pipeline
    
    Args:
        start_date: Start date for data collection
        end_date: End date for data collection
        use_optuna: Use Optuna for hyperparameter tuning
    """
    logger.info("=" * 80)
    logger.info("RUNNING FULL ML PIPELINE")
    logger.info("=" * 80)
    
    start_time = datetime.now()
    
    # Step 1: Collect data
    collect_data(start_date, end_date)
    
    # Step 2: Engineer features
    features_df = engineer_features()
    
    # Step 3: Preprocess data
    preprocess_data(features_df)
    
    # Step 4: Train models
    train_models(use_optuna=use_optuna)
    
    # Step 5: Evaluate models
    best_clf_name, best_reg_name = evaluate_models()
    
    # Step 6: Make predictions
    make_predictions(model_name=best_clf_name, task_type='classification')
    make_predictions(model_name=best_reg_name, task_type='regression')
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("=" * 80)
    logger.info(f"PIPELINE COMPLETED IN {duration}")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BABA Premium/Discount Prediction System')
    parser.add_argument('--step', type=str, choices=['collect', 'features', 'preprocess', 'train', 'evaluate', 'predict', 'full'],
                       default='full', help='Pipeline step to run')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--no-optuna', action='store_true', help='Disable Optuna for hyperparameter tuning')
    parser.add_argument('--model', type=str, default='xgboost', help='Model name for prediction')
    parser.add_argument('--task', type=str, choices=['classification', 'regression'], default='classification',
                       help='Task type for prediction')
    
    args = parser.parse_args()
    
    if args.step == 'collect':
        collect_data(args.start_date, args.end_date)
    elif args.step == 'features':
        engineer_features()
    elif args.step == 'preprocess':
        preprocess_data()
    elif args.step == 'train':
        train_models(use_optuna=not args.no_optuna)
    elif args.step == 'evaluate':
        evaluate_models()
    elif args.step == 'predict':
        make_predictions(args.model, args.task)
    elif args.step == 'full':
        run_full_pipeline(args.start_date, args.end_date, use_optuna=not args.no_optuna)

