"""
Model Training Module - Train and tune ML models
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import logging
import joblib
from datetime import datetime

# ML libraries
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler

import config

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and tune machine learning models"""
    
    def __init__(self, task_type: str = 'classification'):
        """
        Initialize ModelTrainer
        
        Args:
            task_type: 'classification' or 'regression'
        """
        self.task_type = task_type
        self.models = {}
        self.best_params = {}
        self.results = {}
        
    def prepare_data(self, 
                    df: pd.DataFrame,
                    feature_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target from DataFrame
        
        Args:
            df: Input DataFrame
            feature_columns: List of feature column names
            
        Returns:
            Tuple of (X, y)
        """
        X = df[feature_columns].values
        
        if self.task_type == 'classification':
            y = df['target_class'].values
        else:
            y = df['baba_prem_discount'].values
        
        return X, y
    
    def train_xgboost(self,
                     X_train: np.ndarray,
                     y_train: np.ndarray,
                     X_val: np.ndarray,
                     y_val: np.ndarray,
                     use_optuna: bool = True) -> Any:
        """
        Train XGBoost model with hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            use_optuna: Use Optuna for hyperparameter tuning
            
        Returns:
            Trained model
        """
        logger.info(f"Training XGBoost {self.task_type} model")
        
        if use_optuna:
            model = self._tune_xgboost_optuna(X_train, y_train, X_val, y_val)
        else:
            model = self._tune_xgboost_grid(X_train, y_train)
        
        self.models['xgboost'] = model
        return model
    
    def _tune_xgboost_optuna(self,
                            X_train: np.ndarray,
                            y_train: np.ndarray,
                            X_val: np.ndarray,
                            y_val: np.ndarray) -> Any:
        """Tune XGBoost using Optuna"""
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                'random_state': config.RANDOM_STATE
            }
            
            if self.task_type == 'classification':
                params['objective'] = 'binary:logistic'
                params['eval_metric'] = 'auc'
                model = xgb.XGBClassifier(**params)
            else:
                params['objective'] = 'reg:squarederror'
                params['eval_metric'] = 'rmse'
                model = xgb.XGBRegressor(**params)
            
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            if self.task_type == 'classification':
                y_pred = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_pred)
            else:
                y_pred = model.predict(X_val)
                # Calculate RMSE manually for compatibility with older scikit-learn
                mse = mean_squared_error(y_val, y_pred)
                score = -np.sqrt(mse)  # Negative RMSE

            return score
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=config.RANDOM_STATE)
        )
        study.optimize(objective, n_trials=config.OPTUNA_N_TRIALS, timeout=config.OPTUNA_TIMEOUT)
        
        logger.info(f"Best XGBoost params: {study.best_params}")
        self.best_params['xgboost'] = study.best_params
        
        # Train final model with best params
        best_params = study.best_params.copy()
        best_params['random_state'] = config.RANDOM_STATE
        
        if self.task_type == 'classification':
            best_params['objective'] = 'binary:logistic'
            model = xgb.XGBClassifier(**best_params)
        else:
            best_params['objective'] = 'reg:squarederror'
            model = xgb.XGBRegressor(**best_params)
        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        return model
    
    def _tune_xgboost_grid(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Tune XGBoost using GridSearchCV"""
        
        if self.task_type == 'classification':
            model = xgb.XGBClassifier(random_state=config.RANDOM_STATE)
            param_grid = config.XGBOOST_PARAMS_CLASSIFICATION
        else:
            model = xgb.XGBRegressor(random_state=config.RANDOM_STATE)
            param_grid = config.XGBOOST_PARAMS_REGRESSION
        
        # Use RandomizedSearchCV for faster tuning
        search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=50,
            cv=5,
            scoring='roc_auc' if self.task_type == 'classification' else 'neg_root_mean_squared_error',
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )
        
        search.fit(X_train, y_train)
        
        logger.info(f"Best XGBoost params: {search.best_params_}")
        self.best_params['xgboost'] = search.best_params_
        
        return search.best_estimator_
    
    def train_lightgbm(self,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_val: np.ndarray,
                      y_val: np.ndarray,
                      use_optuna: bool = True) -> Any:
        """
        Train LightGBM model with hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            use_optuna: Use Optuna for hyperparameter tuning
            
        Returns:
            Trained model
        """
        logger.info(f"Training LightGBM {self.task_type} model")
        
        if use_optuna:
            model = self._tune_lightgbm_optuna(X_train, y_train, X_val, y_val)
        else:
            model = self._tune_lightgbm_grid(X_train, y_train)
        
        self.models['lightgbm'] = model
        return model
    
    def _tune_lightgbm_optuna(self,
                             X_train: np.ndarray,
                             y_train: np.ndarray,
                             X_val: np.ndarray,
                             y_val: np.ndarray) -> Any:
        """Tune LightGBM using Optuna"""
        
        def objective(trial):
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                'random_state': config.RANDOM_STATE,
                'verbose': -1
            }
            
            if self.task_type == 'classification':
                model = lgb.LGBMClassifier(**params)
            else:
                model = lgb.LGBMRegressor(**params)
            
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)])
            
            if self.task_type == 'classification':
                y_pred = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_pred)
            else:
                y_pred = model.predict(X_val)
                # Calculate RMSE manually for compatibility with older scikit-learn
                mse = mean_squared_error(y_val, y_pred)
                score = -np.sqrt(mse)  # Negative RMSE

            return score
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=config.RANDOM_STATE)
        )
        study.optimize(objective, n_trials=config.OPTUNA_N_TRIALS, timeout=config.OPTUNA_TIMEOUT)
        
        logger.info(f"Best LightGBM params: {study.best_params}")
        self.best_params['lightgbm'] = study.best_params
        
        # Train final model
        best_params = study.best_params.copy()
        best_params['random_state'] = config.RANDOM_STATE
        best_params['verbose'] = -1
        
        if self.task_type == 'classification':
            model = lgb.LGBMClassifier(**best_params)
        else:
            model = lgb.LGBMRegressor(**best_params)
        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)])

        return model

    def _tune_lightgbm_grid(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Tune LightGBM using GridSearchCV"""

        if self.task_type == 'classification':
            model = lgb.LGBMClassifier(random_state=config.RANDOM_STATE, verbose=-1)
            param_grid = config.LIGHTGBM_PARAMS_CLASSIFICATION
        else:
            model = lgb.LGBMRegressor(random_state=config.RANDOM_STATE, verbose=-1)
            param_grid = config.LIGHTGBM_PARAMS_REGRESSION

        search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=50,
            cv=5,
            scoring='roc_auc' if self.task_type == 'classification' else 'neg_root_mean_squared_error',
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )

        search.fit(X_train, y_train)

        logger.info(f"Best LightGBM params: {search.best_params_}")
        self.best_params['lightgbm'] = search.best_params_

        return search.best_estimator_

    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Train Random Forest model"""
        logger.info(f"Training Random Forest {self.task_type} model")

        if self.task_type == 'classification':
            model = RandomForestClassifier(random_state=config.RANDOM_STATE)
            param_grid = config.RF_PARAMS_CLASSIFICATION
        else:
            model = RandomForestRegressor(random_state=config.RANDOM_STATE)
            param_grid = config.RF_PARAMS_REGRESSION

        search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=30,
            cv=5,
            scoring='roc_auc' if self.task_type == 'classification' else 'neg_root_mean_squared_error',
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )

        search.fit(X_train, y_train)

        logger.info(f"Best Random Forest params: {search.best_params_}")
        self.best_params['random_forest'] = search.best_params_
        self.models['random_forest'] = search.best_estimator_

        return search.best_estimator_

    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Train Logistic Regression or Ridge Regression model"""
        logger.info(f"Training {'Logistic Regression' if self.task_type == 'classification' else 'Ridge Regression'} model")

        if self.task_type == 'classification':
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000]
            }
            model = LogisticRegression(random_state=config.RANDOM_STATE)
        else:
            param_grid = {
                'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
            }
            model = Ridge(random_state=config.RANDOM_STATE)

        search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='roc_auc' if self.task_type == 'classification' else 'neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

        search.fit(X_train, y_train)

        logger.info(f"Best params: {search.best_params_}")
        self.best_params['logistic_regression' if self.task_type == 'classification' else 'ridge'] = search.best_params_
        self.models['logistic_regression' if self.task_type == 'classification' else 'ridge'] = search.best_estimator_

        return search.best_estimator_

    def train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Train Neural Network (MLP) model"""
        logger.info(f"Training Neural Network {self.task_type} model")

        if self.task_type == 'classification':
            model = MLPClassifier(random_state=config.RANDOM_STATE, early_stopping=True)
        else:
            model = MLPRegressor(random_state=config.RANDOM_STATE, early_stopping=True)

        search = RandomizedSearchCV(
            model,
            config.NN_PARAMS,
            n_iter=20,
            cv=5,
            scoring='roc_auc' if self.task_type == 'classification' else 'neg_root_mean_squared_error',
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )

        search.fit(X_train, y_train)

        logger.info(f"Best Neural Network params: {search.best_params_}")
        self.best_params['neural_network'] = search.best_params_
        self.models['neural_network'] = search.best_estimator_

        return search.best_estimator_

    def train_all_models(self,
                        train_df: pd.DataFrame,
                        val_df: pd.DataFrame,
                        feature_columns: List[str],
                        use_optuna: bool = True) -> Dict[str, Any]:
        """
        Train all models

        Args:
            train_df: Training data
            val_df: Validation data
            feature_columns: List of feature columns
            use_optuna: Use Optuna for XGBoost and LightGBM

        Returns:
            Dictionary of trained models
        """
        logger.info("Training all models")

        # Prepare data
        X_train, y_train = self.prepare_data(train_df, feature_columns)
        X_val, y_val = self.prepare_data(val_df, feature_columns)

        # Train models
        self.train_xgboost(X_train, y_train, X_val, y_val, use_optuna)
        self.train_lightgbm(X_train, y_train, X_val, y_val, use_optuna)
        self.train_random_forest(X_train, y_train)
        self.train_logistic_regression(X_train, y_train)
        self.train_neural_network(X_train, y_train)

        logger.info(f"Trained {len(self.models)} models")

        return self.models

    def save_models(self, output_dir: str = None):
        """Save all trained models"""
        output_dir = output_dir or config.MODEL_DIR

        for name, model in self.models.items():
            model_path = f"{output_dir}/{name}_{self.task_type}.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} model to {model_path}")

        # Save best params
        params_path = f"{output_dir}/best_params_{self.task_type}.pkl"
        joblib.dump(self.best_params, params_path)
        logger.info(f"Saved best params to {params_path}")


if __name__ == "__main__":
    # Example usage
    import os

    # Load preprocessed data
    train_df = pd.read_parquet(config.TRAIN_DATA_FILE)
    val_df = pd.read_parquet(config.VAL_DATA_FILE)

    # Load feature columns
    feature_columns = joblib.load(f"{config.MODEL_DIR}/feature_columns.pkl")

    # Train classification models
    logger.info("=" * 80)
    logger.info("TRAINING CLASSIFICATION MODELS")
    logger.info("=" * 80)

    clf_trainer = ModelTrainer(task_type='classification')
    clf_models = clf_trainer.train_all_models(train_df, val_df, feature_columns, use_optuna=True)
    clf_trainer.save_models()

    # Train regression models
    logger.info("=" * 80)
    logger.info("TRAINING REGRESSION MODELS")
    logger.info("=" * 80)

    reg_trainer = ModelTrainer(task_type='regression')
    reg_models = reg_trainer.train_all_models(train_df, val_df, feature_columns, use_optuna=True)
    reg_trainer.save_models()

