"""
Model Evaluation Module - Evaluate and compare models
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
import joblib
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
)
import matplotlib.pyplot as plt
import seaborn as sns

import config

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate and compare machine learning models"""
    
    def __init__(self, task_type: str = 'classification'):
        """
        Initialize ModelEvaluator
        
        Args:
            task_type: 'classification' or 'regression'
        """
        self.task_type = task_type
        self.results = {}
        
    def evaluate_classification(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Evaluate classification model
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    def evaluate_regression(self,
                           y_true: np.ndarray,
                           y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate regression model
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
        }
        
        return metrics
    
    def evaluate_model(self,
                      model: Any,
                      X: np.ndarray,
                      y: np.ndarray,
                      model_name: str) -> Dict[str, float]:
        """
        Evaluate a single model
        
        Args:
            model: Trained model
            X: Features
            y: Target
            model_name: Name of the model
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating {model_name} ({self.task_type})")
        
        if self.task_type == 'classification':
            y_pred = model.predict(X)
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X)[:, 1]
            else:
                y_pred_proba = None
            
            metrics = self.evaluate_classification(y, y_pred, y_pred_proba)
            
        else:  # regression
            y_pred = model.predict(X)
            metrics = self.evaluate_regression(y, y_pred)
        
        return metrics
    
    def evaluate_all_models(self,
                           models: Dict[str, Any],
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_val: np.ndarray,
                           y_val: np.ndarray,
                           X_test: np.ndarray,
                           y_test: np.ndarray) -> pd.DataFrame:
        """
        Evaluate all models on train, validation, and test sets
        
        Args:
            models: Dictionary of trained models
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            
        Returns:
            DataFrame with evaluation results
        """
        logger.info(f"Evaluating all {self.task_type} models")
        
        results = []
        
        for model_name, model in models.items():
            # Evaluate on train set
            train_metrics = self.evaluate_model(model, X_train, y_train, model_name)
            train_metrics = {f'train_{k}': v for k, v in train_metrics.items()}
            
            # Evaluate on validation set
            val_metrics = self.evaluate_model(model, X_val, y_val, model_name)
            val_metrics = {f'val_{k}': v for k, v in val_metrics.items()}
            
            # Evaluate on test set
            test_metrics = self.evaluate_model(model, X_test, y_test, model_name)
            test_metrics = {f'test_{k}': v for k, v in test_metrics.items()}
            
            # Combine all metrics
            all_metrics = {'model': model_name}
            all_metrics.update(train_metrics)
            all_metrics.update(val_metrics)
            all_metrics.update(test_metrics)
            
            results.append(all_metrics)
        
        results_df = pd.DataFrame(results)
        self.results = results_df
        
        return results_df
    
    def select_best_model(self, results_df: pd.DataFrame, metric: str = None) -> Tuple[str, Dict]:
        """
        Select the best model based on validation performance
        
        Args:
            results_df: DataFrame with evaluation results
            metric: Metric to use for selection (default: roc_auc for classification, rmse for regression)
            
        Returns:
            Tuple of (best_model_name, best_model_metrics)
        """
        if metric is None:
            if self.task_type == 'classification':
                metric = 'val_roc_auc'
                ascending = False
            else:
                metric = 'val_rmse'
                ascending = True
        else:
            # Determine if higher is better
            ascending = metric in ['val_rmse', 'val_mae', 'val_mape']
        
        # Sort by metric
        sorted_df = results_df.sort_values(metric, ascending=ascending)
        best_row = sorted_df.iloc[0]
        
        best_model_name = best_row['model']
        best_metrics = best_row.to_dict()
        
        logger.info(f"Best model: {best_model_name} with {metric} = {best_row[metric]:.4f}")
        
        return best_model_name, best_metrics
    
    def plot_model_comparison(self, results_df: pd.DataFrame, output_dir: str = None):
        """
        Plot model comparison
        
        Args:
            results_df: DataFrame with evaluation results
            output_dir: Output directory for plots
        """
        output_dir = output_dir or config.RESULTS_DIR
        
        if self.task_type == 'classification':
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        else:
            metrics = ['rmse', 'mae', 'r2', 'mape']
        
        # Filter available metrics
        available_metrics = [m for m in metrics if f'test_{m}' in results_df.columns]
        
        # Create subplots
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(available_metrics):
            ax = axes[i]
            
            # Plot train, val, test for each model
            models = results_df['model'].values
            train_vals = results_df[f'train_{metric}'].values
            val_vals = results_df[f'val_{metric}'].values
            test_vals = results_df[f'test_{metric}'].values
            
            x = np.arange(len(models))
            width = 0.25
            
            ax.bar(x - width, train_vals, width, label='Train', alpha=0.8)
            ax.bar(x, val_vals, width, label='Validation', alpha=0.8)
            ax.bar(x + width, test_vals, width, label='Test', alpha=0.8)
            
            ax.set_xlabel('Model')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plot_path = f"{output_dir}/model_comparison_{self.task_type}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison plot to {plot_path}")
        plt.close()
    
    def plot_feature_importance(self,
                               model: Any,
                               feature_names: List[str],
                               model_name: str,
                               output_dir: str = None,
                               top_n: int = 20):
        """
        Plot feature importance
        
        Args:
            model: Trained model
            feature_names: List of feature names
            model_name: Name of the model
            output_dir: Output directory
            top_n: Number of top features to plot
        """
        output_dir = output_dir or config.RESULTS_DIR
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).flatten()
        else:
            logger.warning(f"Model {model_name} does not have feature importance")
            return
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        # Sort and get top N
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        plot_path = f"{output_dir}/feature_importance_{model_name}_{self.task_type}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {plot_path}")
        plt.close()
        
        # Save to CSV
        csv_path = f"{output_dir}/feature_importance_{model_name}_{self.task_type}.csv"
        importance_df.to_csv(csv_path, index=False)
        logger.info(f"Saved feature importance to {csv_path}")
    
    def save_results(self, results_df: pd.DataFrame, output_dir: str = None):
        """
        Save evaluation results
        
        Args:
            results_df: DataFrame with results
            output_dir: Output directory
        """
        output_dir = output_dir or config.RESULTS_DIR
        
        # Save to CSV
        csv_path = f"{output_dir}/evaluation_results_{self.task_type}.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Saved evaluation results to {csv_path}")
        
        # Save to JSON
        json_path = f"{output_dir}/evaluation_results_{self.task_type}.json"
        results_df.to_json(json_path, orient='records', indent=2)
        logger.info(f"Saved evaluation results to {json_path}")


if __name__ == "__main__":
    # Example usage
    import os
    from model_trainer import ModelTrainer
    
    # Load data
    train_df = pd.read_parquet(config.TRAIN_DATA_FILE)
    val_df = pd.read_parquet(config.VAL_DATA_FILE)
    test_df = pd.read_parquet(config.TEST_DATA_FILE)
    
    # Load feature columns
    feature_columns = joblib.load(f"{config.MODEL_DIR}/feature_columns.pkl")
    
    # Prepare data
    trainer = ModelTrainer(task_type='classification')
    X_train, y_train = trainer.prepare_data(train_df, feature_columns)
    X_val, y_val = trainer.prepare_data(val_df, feature_columns)
    X_test, y_test = trainer.prepare_data(test_df, feature_columns)
    
    # Load models
    models = {}
    for model_name in ['xgboost', 'lightgbm', 'random_forest', 'logistic_regression', 'neural_network']:
        model_path = f"{config.MODEL_DIR}/{model_name}_classification.pkl"
        if os.path.exists(model_path):
            models[model_name] = joblib.load(model_path)
    
    # Evaluate
    evaluator = ModelEvaluator(task_type='classification')
    results_df = evaluator.evaluate_all_models(models, X_train, y_train, X_val, y_val, X_test, y_test)
    
    print(results_df)
    
    # Select best model
    best_model_name, best_metrics = evaluator.select_best_model(results_df)
    
    # Plot comparison
    evaluator.plot_model_comparison(results_df)
    
    # Plot feature importance for best model
    evaluator.plot_feature_importance(models[best_model_name], feature_columns, best_model_name)
    
    # Save results
    evaluator.save_results(results_df)

