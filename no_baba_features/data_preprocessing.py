"""
Data Preprocessing Module - Clean, scale, and split data
"""
import pandas as pd
import numpy as np
from typing import Tuple, List
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import joblib

import config

# Setup logging
logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess data for model training"""
    
    def __init__(self):
        """Initialize DataPreprocessor"""
        self.scaler = None
        self.imputer = None
        self.feature_columns = None
        
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'forward_fill') -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing values
                     ('forward_fill', 'mean', 'median', 'drop')
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info(f"Handling missing values using strategy: {strategy}")
        
        df = df.copy()
        
        # Log missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Missing values before handling:\n{missing_counts[missing_counts > 0]}")
        
        if strategy == 'forward_fill':
            # Forward fill for time series data
            df = df.ffill()
            # Backward fill for any remaining NaNs at the start
            df = df.bfill()

            # If there are still NaNs (e.g., entire columns with many missing values),
            # use median imputation as fallback
            if df.isnull().sum().sum() > 0:
                logger.warning(f"Still have {df.isnull().sum().sum()} NaNs after forward/backward fill. Using median imputation as fallback.")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if df[col].isnull().sum() > 0:
                        median_val = df[col].median()
                        if pd.isna(median_val):
                            # If median is also NaN (all values are NaN), use 0
                            df[col].fillna(0, inplace=True)
                            logger.warning(f"Column {col} has all NaN values, filling with 0")
                        else:
                            df[col].fillna(median_val, inplace=True)

        elif strategy == 'mean':
            # Use mean imputation
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            self.imputer = SimpleImputer(strategy='mean')
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])

        elif strategy == 'median':
            # Use median imputation
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            self.imputer = SimpleImputer(strategy='median')
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])

        elif strategy == 'drop':
            # Drop rows with missing values
            df = df.dropna()

        # Log remaining missing values
        missing_after = df.isnull().sum().sum()
        logger.info(f"Missing values after handling: {missing_after}")

        # Final check - ensure no NaNs remain
        if missing_after > 0:
            logger.error(f"Still have {missing_after} missing values after preprocessing!")
            missing_cols = df.isnull().sum()
            logger.error(f"Columns with missing values:\n{missing_cols[missing_cols > 0]}")
        
        return df
    
    def create_classification_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary classification target (premium vs discount)
        
        Args:
            df: DataFrame with continuous target
            
        Returns:
            DataFrame with classification target added
        """
        logger.info("Creating classification target")
        
        df = df.copy()
        
        # Binary classification: 1 if premium (> 0), 0 if discount (<= 0)
        df['target_class'] = (df['baba_prem_discount'] > config.CLASSIFICATION_THRESHOLD).astype(int)
        
        # Log class distribution
        class_dist = df['target_class'].value_counts()
        logger.info(f"Class distribution:\n{class_dist}")
        logger.info(f"Class balance: {class_dist[1] / len(df) * 100:.2f}% premium")
        
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets (time-series aware)
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Splitting data into train/val/test sets")
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculate split indices
        n = len(df)
        train_end = int(n * config.TRAIN_RATIO)
        val_end = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))
        
        # Split data
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        logger.info(f"Train set: {len(train_df)} samples ({train_df['date'].min()} to {train_df['date'].max()})")
        logger.info(f"Val set: {len(val_df)} samples ({val_df['date'].min()} to {val_df['date'].max()})")
        logger.info(f"Test set: {len(test_df)} samples ({test_df['date'].min()} to {test_df['date'].max()})")
        
        return train_df, val_df, test_df
    
    def scale_features(self, 
                      train_df: pd.DataFrame,
                      val_df: pd.DataFrame,
                      test_df: pd.DataFrame,
                      scaler_type: str = 'standard') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Scale features using StandardScaler or RobustScaler
        
        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            scaler_type: Type of scaler ('standard' or 'robust')
            
        Returns:
            Tuple of scaled (train_df, val_df, test_df)
        """
        logger.info(f"Scaling features using {scaler_type} scaler")
        
        # Identify feature columns (exclude date and targets)
        exclude_cols = ['date', 'baba_prem_discount', 'target_class']
        self.feature_columns = [col for col in train_df.columns if col not in exclude_cols]
        
        # Initialize scaler
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        # Check for NaN values before scaling
        train_nans = train_df[self.feature_columns].isnull().sum().sum()
        val_nans = val_df[self.feature_columns].isnull().sum().sum()
        test_nans = test_df[self.feature_columns].isnull().sum().sum()

        if train_nans > 0 or val_nans > 0 or test_nans > 0:
            logger.error(f"Found NaN values before scaling: train={train_nans}, val={val_nans}, test={test_nans}")
            # Fill any remaining NaNs with 0 as last resort
            logger.warning("Filling remaining NaNs with 0 as last resort")
            train_df[self.feature_columns] = train_df[self.feature_columns].fillna(0)
            val_df[self.feature_columns] = val_df[self.feature_columns].fillna(0)
            test_df[self.feature_columns] = test_df[self.feature_columns].fillna(0)

        # Fit scaler on training data only
        train_df_scaled = train_df.copy()
        val_df_scaled = val_df.copy()
        test_df_scaled = test_df.copy()

        train_df_scaled[self.feature_columns] = self.scaler.fit_transform(train_df[self.feature_columns])
        val_df_scaled[self.feature_columns] = self.scaler.transform(val_df[self.feature_columns])
        test_df_scaled[self.feature_columns] = self.scaler.transform(test_df[self.feature_columns])

        logger.info(f"Scaled {len(self.feature_columns)} features")
        
        return train_df_scaled, val_df_scaled, test_df_scaled
    
    def prepare_data_for_training(self, 
                                  df: pd.DataFrame,
                                  missing_strategy: str = 'forward_fill',
                                  scaler_type: str = 'standard') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Complete preprocessing pipeline
        
        Args:
            df: Input DataFrame with features
            missing_strategy: Strategy for handling missing values
            scaler_type: Type of scaler to use
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Starting data preprocessing pipeline")
        
        # Handle missing values
        df = self.handle_missing_values(df, strategy=missing_strategy)
        
        # Create classification target
        df = self.create_classification_target(df)
        
        # Split data
        train_df, val_df, test_df = self.split_data(df)
        
        # Scale features
        train_df, val_df, test_df = self.scale_features(train_df, val_df, test_df, scaler_type)
        
        logger.info("Data preprocessing completed")
        
        return train_df, val_df, test_df
    
    def save_preprocessor(self, output_dir: str = None):
        """
        Save scaler and imputer for later use
        
        Args:
            output_dir: Output directory
        """
        output_dir = output_dir or config.MODEL_DIR
        
        if self.scaler is not None:
            scaler_path = f"{output_dir}/scaler.pkl"
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Saved scaler to {scaler_path}")
        
        if self.imputer is not None:
            imputer_path = f"{output_dir}/imputer.pkl"
            joblib.dump(self.imputer, imputer_path)
            logger.info(f"Saved imputer to {imputer_path}")
        
        if self.feature_columns is not None:
            features_path = f"{output_dir}/feature_columns.pkl"
            joblib.dump(self.feature_columns, features_path)
            logger.info(f"Saved feature columns to {features_path}")
    
    def load_preprocessor(self, input_dir: str = None):
        """
        Load saved scaler and imputer
        
        Args:
            input_dir: Input directory
        """
        input_dir = input_dir or config.MODEL_DIR
        
        scaler_path = f"{input_dir}/scaler.pkl"
        imputer_path = f"{input_dir}/imputer.pkl"
        features_path = f"{input_dir}/feature_columns.pkl"
        
        import os
        
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")
        
        if os.path.exists(imputer_path):
            self.imputer = joblib.load(imputer_path)
            logger.info(f"Loaded imputer from {imputer_path}")
        
        if os.path.exists(features_path):
            self.feature_columns = joblib.load(features_path)
            logger.info(f"Loaded feature columns from {features_path}")


if __name__ == "__main__":
    # Example usage
    import os
    
    # Load features
    if os.path.exists(config.FEATURES_FILE):
        features_df = pd.read_parquet(config.FEATURES_FILE)
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        train_df, val_df, test_df = preprocessor.prepare_data_for_training(features_df)
        
        # Save preprocessed data
        train_df.to_parquet(config.TRAIN_DATA_FILE, index=False)
        val_df.to_parquet(config.VAL_DATA_FILE, index=False)
        test_df.to_parquet(config.TEST_DATA_FILE, index=False)
        
        # Save preprocessor
        preprocessor.save_preprocessor()
        
        logger.info("Preprocessing completed and data saved")

