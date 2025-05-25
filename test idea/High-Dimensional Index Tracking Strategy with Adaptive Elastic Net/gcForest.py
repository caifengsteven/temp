import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import Bloomberg API
try:
    import pdblp
    BLOOMBERG_AVAILABLE = True
    print("Bloomberg API available!")
    # Initialize Bloomberg connection with longer timeout
    con = pdblp.BCon(debug=False, port=8194, timeout=60000)  # Increased timeout
    con.start()
except ImportError:
    BLOOMBERG_AVAILABLE = False
    print("Bloomberg API not available. Will use synthetic data.")

class CascadeForest:
    """
    Implementation of cascade forest with configurable architecture
    """
    def __init__(self, n_estimators=500, n_cascades=5, n_classes=None, 
                 use_custom_features=True, feature_importance=True, random_state=42):
        self.n_estimators = n_estimators
        self.n_cascades = n_cascades
        self.n_classes = n_classes
        self.use_custom_features = use_custom_features
        self.feature_importance = feature_importance
        self.random_state = random_state
        self.rf_models = []
        self.et_models = []
        self.feature_importances_ = None
        
    def fit(self, X, y):
        """
        Fit the cascade forest to training data
        """
        print(f"Training CascadeForest with {self.n_cascades} cascade levels...")
        if self.n_classes is None:
            self.n_classes = len(np.unique(y))
        
        n_samples, n_features = X.shape
        print(f"Input shape: {X.shape}")
        
        # Ensure no NaN or Inf values
        X_train = X.copy()
        
        # Store individual feature importances
        all_importances = np.zeros(n_features)
        
        # Train each level of the cascade
        for i in range(self.n_cascades):
            print(f"Training cascade level {i+1}...")
            
            # For each cascade level, we use two types of forests
            rf = RandomForestClassifier(
                n_estimators=self.n_estimators, 
                max_features='sqrt',  # Good for financial data
                random_state=self.random_state
            )
            
            et = ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                max_features='sqrt',  # Good for financial data  
                random_state=self.random_state
            )
            
            # Train the models on current features
            rf.fit(X_train, y)
            et.fit(X_train, y)
            
            # Collect feature importance (only for original features)
            if i == 0 and self.feature_importance:
                all_importances = rf.feature_importances_
                
            # Get predictions to use as features for next level
            rf_preds = rf.predict_proba(X_train)
            et_preds = et.predict_proba(X_train)
            
            # Store the models
            self.rf_models.append(rf)
            self.et_models.append(et)
            
            # For the final level, we don't need to augment features
            if i < self.n_cascades - 1:
                # Combine original features with model predictions for next level
                if self.use_custom_features and i == 0:
                    # Add custom combination features for financial data
                    X_custom = self._create_custom_financial_features(X_train, rf_preds, et_preds)
                    X_train = np.hstack([X_train, X_custom])
                else:
                    # Regular feature augmentation
                    X_train = np.hstack([X_train, rf_preds, et_preds])
                
                print(f"New training shape: {X_train.shape}")
        
        # Store feature importances
        if self.feature_importance:
            self.feature_importances_ = all_importances
            
        return self
    
    def predict(self, X):
        """
        Predict class labels for X
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X
        """
        if not self.rf_models:
            raise ValueError("Model hasn't been trained yet")
        
        X_test = X.copy()
        
        # Process through each level of the cascade
        for i in range(self.n_cascades):
            rf = self.rf_models[i]
            et = self.et_models[i]
            
            rf_preds = rf.predict_proba(X_test)
            et_preds = et.predict_proba(X_test)
            
            # For the final level, return the average predictions
            if i == self.n_cascades - 1:
                return (rf_preds + et_preds) / 2
            
            # Otherwise, augment features for the next level
            if self.use_custom_features and i == 0:
                # Add custom combination features for financial data
                X_custom = self._create_custom_financial_features(X_test, rf_preds, et_preds)
                X_test = np.hstack([X_test, X_custom])
            else:
                # Regular feature augmentation
                X_test = np.hstack([X_test, rf_preds, et_preds])
        
        # Should not reach here
        return None
    
    def _create_custom_financial_features(self, X, rf_preds, et_preds):
        """
        Create custom features specific to financial data
        
        Parameters:
        -----------
        X : ndarray
            Original features
        rf_preds : ndarray
            Random Forest predictions
        et_preds : ndarray
            Extra Trees predictions
            
        Returns:
        --------
        ndarray
            Custom features for financial data
        """
        # Combine model predictions (weighted average for different confidence levels)
        avg_preds = (rf_preds * 0.6 + et_preds * 0.4)
        
        # Calculate prediction confidence (max probability)
        confidence = np.max(avg_preds, axis=1).reshape(-1, 1)
        
        # Calculate prediction entropy (uncertainty)
        non_zero_preds = np.clip(avg_preds, 1e-10, 1.0)
        entropy = -np.sum(non_zero_preds * np.log(non_zero_preds), axis=1).reshape(-1, 1)
        
        # Create interaction features (for the most important original features)
        if X.shape[1] > 5:
            # Use top 5 features
            top_indices = np.argsort(np.var(X, axis=0))[-5:]
            interactions = X[:, top_indices[0]].reshape(-1, 1) * X[:, top_indices[1:]].reshape(-1, 4)
        else:
            interactions = np.zeros((X.shape[0], 1))
        
        # Combine all custom features
        return np.hstack([avg_preds, confidence, entropy, interactions])

def fetch_bloomberg_market_data(longer_history=False):
    """
    Fetch market data from Bloomberg if available, otherwise generate synthetic data
    
    Parameters:
    -----------
    longer_history : bool
        Whether to fetch a longer history (useful for training)
    
    Returns:
    --------
    tuple
        X, y, feature_names
    """
    if BLOOMBERG_AVAILABLE:
        try:
            # Define parameters
            if longer_history:
                start_date = '20100101'  # Longer history
            else:
                start_date = '20180101'  # More recent data
                
            end_date = datetime.now().strftime('%Y%m%d')
            
            # Define tickers and fields
            tickers = [
                'SPX Index',    # S&P 500
                'INDU Index',   # Dow Jones
                'NDX Index',    # Nasdaq 100
                'RTY Index',    # Russell 2000
                'VIX Index',    # Volatility Index
                'USGG10YR Index',  # 10Y Treasury
                'CL1 Comdty',   # WTI Crude Oil
                'XAU Curncy',   # Gold
                'DXY Index',    # US Dollar Index
                'EURUSD Curncy' # EUR/USD
            ]
            
            # Define fields to request - keeping it simple for robustness
            fields = [
                'PX_LAST',      # Last price
                'PX_VOLUME',    # Volume
                'PX_HIGH',      # High price
                'PX_LOW'        # Low price
            ]
            
            print(f"Fetching data from Bloomberg for {len(tickers)} tickers...")
            print(f"Time period: {start_date} to {end_date}")
            
            # Request data with higher timeout
            data = con.bdh(
                tickers=tickers,
                flds=fields,
                start_date=start_date,
                end_date=end_date,
                longdata=True  # Return long format data
            )
            
            if data is None or data.empty:
                raise ValueError("Bloomberg returned empty dataset")
                
            print(f"Received {len(data)} rows of Bloomberg data")
            print("Processing Bloomberg data...")
            
            # Ensure data has the correct structure
            required_cols = ['ticker', 'field', 'date', 'value']
            for col in required_cols:
                if col not in data.columns:
                    raise ValueError(f"Bloomberg data missing required column: {col}")
            
            # Make sure date is datetime
            data['date'] = pd.to_datetime(data['date'])
            
            # Sort by date for consistency
            data = data.sort_values('date')
            
            # Get unique dates
            unique_dates = data['date'].unique()
            print(f"Data spans {len(unique_dates)} unique dates")
            
            # Focus on the S&P 500 for targets
            spx_data = data[(data['ticker'] == 'SPX Index') & (data['field'] == 'PX_LAST')].copy()
            spx_data = spx_data.sort_values('date')
            
            if len(spx_data) < 20:
                raise ValueError("Insufficient SPX data points")
            
            # Create a continuous series for SPX to handle any missing dates
            date_range = pd.date_range(start=spx_data['date'].min(), end=spx_data['date'].max(), freq='B')
            spx_series = pd.Series(index=date_range)
            spx_series.loc[spx_data['date']] = spx_data['value'].values
            
            # Fill any gaps with forward fill (last available price)
            spx_series = spx_series.fillna(method='ffill')
            
            # Calculate future returns
            future_returns = spx_series.pct_change(periods=5).shift(-5)
            
            # Create target labels
            targets = pd.Series(0, index=future_returns.index)  # Default to neutral
            targets[future_returns > 0.01] = 1  # Up
            targets[future_returns < -0.01] = 2  # Down
            
            # Create a DataFrame to store feature vectors
            feature_df = pd.DataFrame(index=unique_dates)
            
            # Process the tickers and fields
            for ticker in tickers:
                ticker_data = data[data['ticker'] == ticker]
                
                for field in fields:
                    field_data = ticker_data[ticker_data['field'] == field]
                    
                    if not field_data.empty:
                        # Create a series with date as index
                        series = pd.Series(field_data['value'].values, index=field_data['date'])
                        
                        # Add to feature DataFrame
                        feature_df[f"{ticker}_{field}"] = series
            
            # Forward fill missing values within each column
            feature_df = feature_df.fillna(method='ffill')
            
            # Drop rows that still have any NaN values
            feature_df = feature_df.dropna()
            
            # Only keep rows with a corresponding target
            common_dates = feature_df.index.intersection(targets.index)
            feature_df = feature_df.loc[common_dates]
            targets = targets.loc[common_dates]
            
            print(f"Final dataset has {len(feature_df)} samples after date alignment")
            
            # Convert to numpy arrays
            X = feature_df.values
            y = targets.values
            feature_names = feature_df.columns.tolist()
            
            # Final cleaning and preprocessing
            # Replace any remaining NaN or inf values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Check for valid data in X
            if not np.isfinite(X).all():
                print("Warning: Data still contains non-finite values after cleaning")
                print("Applying more aggressive cleaning...")
                X = np.clip(X, -1e10, 1e10)  # Clip extreme values
                X = np.nan_to_num(X)  # Convert remaining NaNs or infs
            
            print(f"Preprocessed dataset: X shape={X.shape}, y shape={y.shape}")
            print(f"Class distribution: {np.bincount(y)}")
            
            return X, y, feature_names
            
        except Exception as e:
            print(f"Error fetching/processing Bloomberg data: {e}")
            print("Falling back to synthetic data...")
    
    # Fall back to synthetic data
    return generate_synthetic_data()

def preprocess_data(X, y, normalize=True, random_state=42):
    """
    Perform data preprocessing and cleaning
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix
    y : ndarray
        Target vector
    normalize : bool
        Whether to normalize features
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X_cleaned, y
    """
    print("Preprocessing data...")
    
    # Check for NaN or infinite values
    if np.isnan(X).any() or np.isinf(X).any():
        print("Cleaning non-finite values...")
        
        # Use imputer to replace NaN values
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        
        # Replace any infinite values with large but finite values
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Clip extreme values
        percentile_99 = np.percentile(X, 99, axis=0)
        percentile_01 = np.percentile(X, 1, axis=0)
        for j in range(X.shape[1]):
            X[:, j] = np.clip(X[:, j], percentile_01[j], percentile_99[j])
    
    # Normalize features if requested
    if normalize:
        print("Normalizing features...")
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    return X, y

def generate_synthetic_data(n_samples=1000, n_features=40, n_classes=3, random_state=42):
    """
    Generate synthetic financial data for testing
    """
    np.random.seed(random_state)
    
    # Generate feature data with correlations resembling financial data
    # First, create a random covariance matrix (some features are correlated)
    cov = np.random.rand(n_features, n_features) * 0.1
    cov = cov @ cov.T + np.diag(np.random.rand(n_features) * 0.9 + 0.1)
    
    # Generate the features
    X = np.random.multivariate_normal(mean=np.zeros(n_features), cov=cov, size=n_samples)
    
    # Generate a target variable that depends on the features
    weights = np.random.randn(n_features) * 0.1
    prob = 1.0 / (1.0 + np.exp(-X.dot(weights)))
    
    # Add some noise to make it more realistic
    noise = np.random.randn(n_samples) * 0.1
    prob += noise
    
    # Convert to classes
    y = np.zeros(n_samples, dtype=int)
    thresholds = np.percentile(prob, [100/n_classes * i for i in range(1, n_classes)])
    
    for i in range(1, n_classes):
        y[prob > thresholds[i-1]] = i
    
    # Generate feature names
    feature_types = ['PX_LAST', 'PX_VOLUME', 'PX_HIGH', 'PX_LOW']
    
    tickers = ['SPX Index', 'INDU Index', 'NDX Index', 'RTY Index', 'VIX Index', 
              'USGG10YR Index', 'CL1 Comdty', 'XAU Curncy', 'DXY Index', 'EURUSD Curncy']
    
    feature_names = []
    for i in range(n_features):
        ticker = tickers[i % len(tickers)]
        feat_type = feature_types[i % len(feature_types)]
        feature_names.append(f"{ticker}_{feat_type}")
    
    print(f"Generated synthetic financial data with {n_samples} samples, {n_features} features, and {n_classes} classes")
    print(f"Class distribution: {np.bincount(y)}")
    return X, y, feature_names

def run_experiment(use_bloomberg=True, test_size=0.2, random_state=42):
    """
    Run the experiment comparing CascadeForest with RandomForest
    
    Parameters:
    -----------
    use_bloomberg : bool
        Whether to try using Bloomberg data
    test_size : float
        Size of the test set
    random_state : int
        Random seed
    """
    # Get data
    print("Loading data...")
    if use_bloomberg and BLOOMBERG_AVAILABLE:
        X, y, feature_names = fetch_bloomberg_market_data(longer_history=True)
    else:
        X, y, feature_names = generate_synthetic_data()
    
    # Verify data consistency
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    if len(X) != len(y):
        print("WARNING: Inconsistent lengths of X and y. Falling back to synthetic data.")
        X, y, feature_names = generate_synthetic_data()
    
    # Clean and preprocess data
    X, y = preprocess_data(X, y, normalize=True, random_state=random_state)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train and evaluate CascadeForest
    print("\nTraining CascadeForest...")
    start_time = time.time()
    model = CascadeForest(
        n_estimators=100,  # Use fewer trees for faster training
        n_cascades=3,      # Use fewer cascade levels
        n_classes=len(np.unique(y)),
        use_custom_features=True,
        feature_importance=True,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    training_time = time.time() - start_time
    
    print(f"CascadeForest Accuracy: {accuracy:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('CascadeForest Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('cascade_forest_confusion_matrix.png')
    plt.close()
    
    # Compare with RandomForest
    print("\nTraining RandomForest for comparison...")
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=random_state)
    rf.fit(X_train, y_train)
    
    # Make predictions
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)
    
    # Calculate accuracy
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_training_time = time.time() - start_time
    
    print(f"RandomForest Accuracy: {rf_accuracy:.4f}")
    print(f"Training Time: {rf_training_time:.2f} seconds")
    
    # Plot feature importance
    if feature_names:
        plt.figure(figsize=(12, 8))
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot the top features
        top_n = min(20, len(feature_names))
        plt.title('Random Forest Feature Importances')
        plt.bar(range(top_n), importances[indices[:top_n]], color='b', align='center')
        plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
    
    # Plot ROC curve for each class
    plt.figure(figsize=(10, 8))
    for i in range(len(np.unique(y))):
        y_true_binary = (y_test == i).astype(int)
        
        # ROC curve for CascadeForest
        fpr_cf, tpr_cf, _ = roc_curve(y_true_binary, y_proba[:, i])
        roc_auc_cf = auc(fpr_cf, tpr_cf)
        plt.plot(fpr_cf, tpr_cf, lw=2, 
                 label=f'Class {i} - CascadeForest (AUC = {roc_auc_cf:.2f})')
        
        # ROC curve for RandomForest
        fpr_rf, tpr_rf, _ = roc_curve(y_true_binary, rf_proba[:, i])
        roc_auc_rf = auc(fpr_rf, tpr_rf)
        plt.plot(fpr_rf, tpr_rf, lw=2, linestyle='--',
                 label=f'Class {i} - RandomForest (AUC = {roc_auc_rf:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig('roc_curves.png')
    plt.close()
    
    # Print comparison summary
    print("\nComparison Summary:")
    print("-" * 50)
    print(f"CascadeForest Accuracy: {accuracy:.4f}")
    print(f"RandomForest Accuracy: {rf_accuracy:.4f}")
    print(f"Difference: {abs(accuracy - rf_accuracy):.4f}")
    
    if accuracy > rf_accuracy:
        print(f"CascadeForest outperformed RandomForest by {accuracy - rf_accuracy:.4f}")
    elif rf_accuracy > accuracy:
        print(f"RandomForest outperformed CascadeForest by {rf_accuracy - accuracy:.4f}")
    else:
        print("Both models had identical accuracy")
    
    # Save predictions to CSV for further analysis
    results_df = pd.DataFrame({
        'true_label': y_test,
        'cascade_pred': y_pred,
        'rf_pred': rf_pred
    })
    
    for i in range(len(np.unique(y))):
        results_df[f'cascade_prob_class_{i}'] = y_proba[:, i]
        results_df[f'rf_prob_class_{i}'] = rf_proba[:, i]
    
    results_df.to_csv('prediction_results.csv', index=False)
    print("\nExperiment completed successfully! Results saved.")

if __name__ == "__main__":
    print("Enhanced Deep Forest Implementation for Financial Markets")
    print("=" * 70)
    
    # Run with Bloomberg if available, otherwise use synthetic data
    run_experiment(use_bloomberg=True)