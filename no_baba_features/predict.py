"""
Prediction Module - Make predictions on new data
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
import joblib
from datetime import datetime, timedelta

import config
from data_collector import DataCollector
from csv_processor import CSVProcessor
from feature_engineering import FeatureEngineer
from data_preprocessing import DataPreprocessor

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class Predictor:
    """Make predictions using trained models"""
    
    def __init__(self, model_path: str, task_type: str = 'classification'):
        """
        Initialize Predictor
        
        Args:
            model_path: Path to trained model
            task_type: 'classification' or 'regression'
        """
        self.task_type = task_type
        self.model = joblib.load(model_path)
        
        # Load preprocessor components
        self.scaler = joblib.load(f"{config.MODEL_DIR}/scaler.pkl")
        self.feature_columns = joblib.load(f"{config.MODEL_DIR}/feature_columns.pkl")
        
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Model expects {len(self.feature_columns)} features")
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for prediction
        
        Args:
            df: DataFrame with features
            
        Returns:
            Scaled feature array
        """
        # Ensure all required features are present
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Add missing features with NaN
            for feature in missing_features:
                df[feature] = np.nan
        
        # Select and order features
        X = df[self.feature_columns].values
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Making predictions for {len(df)} samples")
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Make predictions
        if self.task_type == 'classification':
            predictions = self.model.predict(X)
            
            # Get probabilities if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X)[:, 1]
            else:
                probabilities = None
            
            # Create results DataFrame
            results = df[['date']].copy()
            results['predicted_class'] = predictions
            results['predicted_label'] = results['predicted_class'].map({0: 'Discount', 1: 'Premium'})
            
            if probabilities is not None:
                results['probability_premium'] = probabilities
                results['probability_discount'] = 1 - probabilities
        
        else:  # regression
            predictions = self.model.predict(X)
            
            # Create results DataFrame
            results = df[['date']].copy()
            results['predicted_prem_discount'] = predictions
            results['predicted_prem_discount_pct'] = predictions * 100
        
        logger.info("Predictions completed")
        
        return results
    
    def predict_latest(self, 
                      data_collector: DataCollector = None,
                      csv_processor: CSVProcessor = None,
                      feature_engineer: FeatureEngineer = None) -> pd.DataFrame:
        """
        Make predictions for the latest available data
        
        Args:
            data_collector: DataCollector instance (optional)
            csv_processor: CSVProcessor instance (optional)
            feature_engineer: FeatureEngineer instance (optional)
            
        Returns:
            DataFrame with predictions
        """
        logger.info("Collecting latest data for prediction")
        
        # Initialize components if not provided
        if data_collector is None:
            # Get data for the last 60 days to ensure enough history
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
            data_collector = DataCollector(start_date=start_date, end_date=end_date)
        
        if csv_processor is None:
            csv_processor = CSVProcessor()
        
        if feature_engineer is None:
            feature_engineer = FeatureEngineer()
        
        # Collect data
        data = data_collector.collect_all_data()
        
        # Process intraday features
        with csv_processor:
            data['intraday_features'] = csv_processor.process_all_intraday_features()
        
        # Create features
        features_df = feature_engineer.create_all_features(data)
        
        # Get the latest row
        latest_df = features_df.tail(1)
        
        # Make prediction
        predictions = self.predict(latest_df)
        
        return predictions
    
    def backtest_predictions(self,
                           test_df: pd.DataFrame,
                           actual_column: str = None) -> pd.DataFrame:
        """
        Backtest predictions against actual values
        
        Args:
            test_df: Test DataFrame with features and actual values
            actual_column: Column name for actual values
            
        Returns:
            DataFrame with predictions and actual values
        """
        logger.info("Backtesting predictions")
        
        # Make predictions
        predictions = self.predict(test_df)
        
        # Add actual values
        if actual_column is None:
            if self.task_type == 'classification':
                actual_column = 'target_class'
            else:
                actual_column = 'baba_prem_discount'
        
        if actual_column in test_df.columns:
            predictions['actual'] = test_df[actual_column].values
            
            if self.task_type == 'classification':
                predictions['correct'] = (predictions['predicted_class'] == predictions['actual'])
                accuracy = predictions['correct'].mean()
                logger.info(f"Backtest accuracy: {accuracy:.4f}")
            else:
                predictions['error'] = predictions['predicted_prem_discount'] - predictions['actual']
                predictions['abs_error'] = np.abs(predictions['error'])
                mae = predictions['abs_error'].mean()
                logger.info(f"Backtest MAE: {mae:.6f}")
        
        return predictions
    
    def save_predictions(self, predictions: pd.DataFrame, output_path: str = None):
        """
        Save predictions to file
        
        Args:
            predictions: DataFrame with predictions
            output_path: Output file path
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"{config.RESULTS_DIR}/predictions_{self.task_type}_{timestamp}.csv"
        
        predictions.to_csv(output_path, index=False)
        logger.info(f"Saved predictions to {output_path}")


class EnsemblePredictor:
    """Make predictions using ensemble of models"""
    
    def __init__(self, model_paths: Dict[str, str], task_type: str = 'classification'):
        """
        Initialize EnsemblePredictor
        
        Args:
            model_paths: Dictionary of model names and paths
            task_type: 'classification' or 'regression'
        """
        self.task_type = task_type
        self.models = {}
        
        # Load all models
        for name, path in model_paths.items():
            self.models[name] = joblib.load(path)
            logger.info(f"Loaded {name} model from {path}")
        
        # Load preprocessor components
        self.scaler = joblib.load(f"{config.MODEL_DIR}/scaler.pkl")
        self.feature_columns = joblib.load(f"{config.MODEL_DIR}/feature_columns.pkl")
    
    def predict(self, df: pd.DataFrame, method: str = 'voting') -> pd.DataFrame:
        """
        Make ensemble predictions
        
        Args:
            df: DataFrame with features
            method: Ensemble method ('voting' or 'averaging')
            
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Making ensemble predictions using {method} method")
        
        # Prepare features
        X = df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        
        if self.task_type == 'classification':
            # Collect predictions from all models
            all_predictions = []
            all_probabilities = []
            
            for name, model in self.models.items():
                pred = model.predict(X_scaled)
                all_predictions.append(pred)
                
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X_scaled)[:, 1]
                    all_probabilities.append(prob)
            
            # Ensemble predictions
            if method == 'voting':
                # Majority voting
                predictions = np.round(np.mean(all_predictions, axis=0)).astype(int)
            else:  # averaging probabilities
                if all_probabilities:
                    avg_prob = np.mean(all_probabilities, axis=0)
                    predictions = (avg_prob > 0.5).astype(int)
                else:
                    predictions = np.round(np.mean(all_predictions, axis=0)).astype(int)
            
            # Create results
            results = df[['date']].copy()
            results['predicted_class'] = predictions
            results['predicted_label'] = results['predicted_class'].map({0: 'Discount', 1: 'Premium'})
            
            if all_probabilities:
                results['probability_premium'] = np.mean(all_probabilities, axis=0)
                results['probability_discount'] = 1 - results['probability_premium']
        
        else:  # regression
            # Collect predictions from all models
            all_predictions = []
            
            for name, model in self.models.items():
                pred = model.predict(X_scaled)
                all_predictions.append(pred)
            
            # Average predictions
            predictions = np.mean(all_predictions, axis=0)
            
            # Create results
            results = df[['date']].copy()
            results['predicted_prem_discount'] = predictions
            results['predicted_prem_discount_pct'] = predictions * 100
        
        logger.info("Ensemble predictions completed")
        
        return results


if __name__ == "__main__":
    # Example usage
    import os
    
    # Load test data
    test_df = pd.read_parquet(config.TEST_DATA_FILE)
    
    # Classification prediction
    logger.info("=" * 80)
    logger.info("CLASSIFICATION PREDICTIONS")
    logger.info("=" * 80)
    
    clf_model_path = f"{config.MODEL_DIR}/xgboost_classification.pkl"
    if os.path.exists(clf_model_path):
        clf_predictor = Predictor(clf_model_path, task_type='classification')
        clf_predictions = clf_predictor.backtest_predictions(test_df)
        clf_predictor.save_predictions(clf_predictions)
        print(clf_predictions.head(10))
    
    # Regression prediction
    logger.info("=" * 80)
    logger.info("REGRESSION PREDICTIONS")
    logger.info("=" * 80)
    
    reg_model_path = f"{config.MODEL_DIR}/xgboost_regression.pkl"
    if os.path.exists(reg_model_path):
        reg_predictor = Predictor(reg_model_path, task_type='regression')
        reg_predictions = reg_predictor.backtest_predictions(test_df)
        reg_predictor.save_predictions(reg_predictions)
        print(reg_predictions.head(10))

