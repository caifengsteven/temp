import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from optuna import Trial, create_study
from optuna.samplers import TPESampler
import optuna
import keras_tuner as kt
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Function to generate simulated financial data
def generate_simulated_data(n_samples=4000, n_features=100, n_lags=5, noise_level=0.2, trend_strength=0.01):
    """
    Generate simulated financial data with trends, seasonality, and noise.
    
    Parameters:
    - n_samples: Number of trading days
    - n_features: Number of financial features
    - n_lags: Number of time lags to include as features
    - noise_level: Level of randomness in the data
    - trend_strength: Strength of the underlying trend
    
    Returns:
    - data: DataFrame with simulated financial data
    - target: Binary target variable (1 if price increased after 5 days, 0 otherwise)
    """
    # Generate base features with correlations
    corr_matrix = np.random.randn(n_features, n_features)
    corr_matrix = np.dot(corr_matrix, corr_matrix.T)
    corr_matrix = (corr_matrix - np.min(corr_matrix)) / (np.max(corr_matrix) - np.min(corr_matrix))
    
    # Generate random data with correlations
    features = np.random.multivariate_normal(mean=np.zeros(n_features), cov=corr_matrix, size=n_samples)
    
    # Add trend component
    trend = np.arange(n_samples) * trend_strength
    trend = np.tile(trend.reshape(-1, 1), (1, n_features))
    features += trend
    
    # Add seasonality (weekly pattern)
    seasonality = np.sin(np.arange(n_samples) * 2 * np.pi / 5)  # 5-day cycle
    seasonality = np.tile(seasonality.reshape(-1, 1), (1, n_features))
    features += seasonality * 0.1
    
    # Create a main "price" feature that others will be correlated with
    price = features[:, 0] * 2 + np.cumsum(np.random.normal(0, 0.05, n_samples))
    
    # Create DataFrame
    data = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(n_features)])
    data['price'] = price
    
    # Add lagged features (time-derived features mentioned in the paper)
    for lag in range(1, n_lags + 1):
        for col in data.columns:
            data[f'{col}_lag_{lag}'] = data[col].shift(lag)
    
    # Add moving averages
    for window in [5, 10, 20]:
        for col in ['price'] + [f'feature_{i}' for i in range(5)]:  # Only for some features to avoid explosion
            data[f'{col}_ma_{window}'] = data[col].rolling(window=window).mean()
    
    # Create binary target: 1 if price goes up in 5 days, 0 otherwise
    target = (data['price'].shift(-5) > data['price']).astype(int)
    
    # Drop NaN values
    data = data.iloc[max(n_lags, 20):]
    target = target.iloc[max(n_lags, 20):]
    
    return data, target

# Function to preprocess data
def preprocess_data(data, target, remove_time_derived=True, apply_pca=False, n_components=None):
    """
    Preprocess the data as described in the paper.
    
    Parameters:
    - data: DataFrame with financial features
    - target: Target variable
    - remove_time_derived: Whether to remove time-derived features
    - apply_pca: Whether to apply PCA
    - n_components: Number of components to keep after PCA
    
    Returns:
    - X_train, X_val, X_test: Processed feature sets
    - y_train, y_val, y_test: Processed target sets
    """
    # Split the data chronologically: 70% train, 20% validation, 10% test
    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.2)
    
    X_train = data.iloc[:train_size]
    y_train = target.iloc[:train_size]
    
    X_val = data.iloc[train_size:train_size+val_size]
    y_val = target.iloc[train_size:train_size+val_size]
    
    X_test = data.iloc[train_size+val_size:]
    y_test = target.iloc[train_size+val_size:]
    
    # Remove time-derived features if specified
    if remove_time_derived:
        time_derived_columns = [col for col in data.columns if ('lag' in col) or ('ma' in col)]
        X_train = X_train.drop(columns=time_derived_columns)
        X_val = X_val.drop(columns=time_derived_columns)
        X_test = X_test.drop(columns=time_derived_columns)
    
    # Normalize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA if specified
    if apply_pca:
        if n_components is None:
            # If not specified, use enough components to explain 95% of variance
            pca = PCA(n_components=0.95)
            pca.fit(X_train_scaled)
        else:
            pca = PCA(n_components=n_components)
            pca.fit(X_train_scaled)
            
        X_train_scaled = pca.transform(X_train_scaled)
        X_val_scaled = pca.transform(X_val_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
        
        # Plot explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Scree Plot')
        plt.grid(True)
        plt.show()
        
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train.values, y_val.values, y_test.values

# 1. MLP (Feedforward Neural Network) Model
def create_mlp_model(params, input_dim):
    """Create a Multi-Layer Perceptron model with given hyperparameters."""
    model = Sequential()
    
    # Input layer
    model.add(Dense(params['units'], activation=params['activation'], input_shape=(input_dim,)))
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout']))
    
    # Hidden layers
    for _ in range(params['n_layers'] - 1):
        model.add(Dense(params['units'], activation=params['activation']))
        model.add(BatchNormalization())
        model.add(Dropout(params['dropout']))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=params['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 2. 1D CNN Model
def create_1d_cnn_model(params, input_shape):
    """Create a 1D CNN model with given hyperparameters."""
    model = Sequential()
    
    # First Conv layer
    model.add(Conv1D(filters=params['filters'],
                     kernel_size=params['kernel_size'],
                     activation=params['activation'],
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(params['dropout']))
    
    # Additional Conv layers
    for _ in range(params['n_conv_layers'] - 1):
        model.add(Conv1D(filters=params['filters'],
                         kernel_size=params['kernel_size'],
                         activation=params['activation']))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(params['dropout']))
    
    # Flatten and Dense layers
    model.add(Flatten())
    model.add(Dense(params['dense_units'], activation=params['activation']))
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout']))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=params['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 3. LSTM (RNN) Model
def create_lstm_model(params, input_shape):
    """Create an LSTM model with given hyperparameters."""
    model = Sequential()
    
    # First LSTM layer
    return_sequences = params['n_lstm_layers'] > 1
    model.add(LSTM(units=params['lstm_units'],
                   return_sequences=return_sequences,
                   input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout']))
    
    # Additional LSTM layers
    for i in range(params['n_lstm_layers'] - 1):
        return_sequences = i < params['n_lstm_layers'] - 2
        model.add(LSTM(units=params['lstm_units'], return_sequences=return_sequences))
        model.add(BatchNormalization())
        model.add(Dropout(params['dropout']))
    
    # Dense layers
    model.add(Dense(params['dense_units'], activation=params['activation']))
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout']))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=params['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Function to prepare data for sequence models
def prepare_sequences(X, y, seq_length):
    """
    Prepare sequences for RNN and CNN models.
    
    Parameters:
    - X: Feature matrix
    - y: Target vector
    - seq_length: Length of sequences
    
    Returns:
    - X_seq: 3D array of shape (n_samples, seq_length, n_features)
    - y_seq: Target vector aligned with X_seq
    """
    n_samples, n_features = X.shape
    X_seq = []
    y_seq = []
    
    for i in range(n_samples - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    
    return np.array(X_seq), np.array(y_seq)

# Function to evaluate model performance
def evaluate_model(model, X, y, threshold=0.5):
    """
    Evaluate model performance with multiple metrics.
    
    Parameters:
    - model: Trained model
    - X: Feature data
    - y: True labels
    - threshold: Classification threshold
    
    Returns:
    - Dictionary of performance metrics
    """
    y_pred_proba = model.predict(X, verbose=0).flatten()
    y_pred = (y_pred_proba > threshold).astype(int)
    
    acc = accuracy_score(y, y_pred)
    bacc = balanced_accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_pred_proba)
    
    return {
        'acc': acc,
        'bacc': bacc,
        'f1': f1,
        'auc': auc
    }

# Function to train a model multiple times with different random seeds
def train_and_evaluate_multiple(model_func, params, X_train, y_train, X_val, y_val, 
                                n_runs=15, epochs=80, batch_size=32, patience=10, is_sequence=False):
    """
    Train a model multiple times with different random seeds and average the results.
    
    Parameters:
    - model_func: Function to create the model
    - params: Model hyperparameters
    - X_train, y_train: Training data
    - X_val, y_val: Validation data
    - n_runs: Number of training runs with different seeds
    - epochs: Maximum number of training epochs
    - batch_size: Batch size for training
    - patience: Patience for early stopping
    - is_sequence: Whether the data should be treated as sequences
    
    Returns:
    - Average and standard deviation of performance metrics across runs
    """
    results = []
    
    for i in range(n_runs):
        # Set different random seed for each run
        tf.random.set_seed(42 + i)
        np.random.seed(42 + i)
        
        # Create model
        if is_sequence:
            input_shape = X_train.shape[1:]
            model = model_func(params, input_shape)
        else:
            input_dim = X_train.shape[1]
            model = model_func(params, input_dim)
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        
        # Train the model
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate on validation data
        metrics = evaluate_model(model, X_val, y_val)
        results.append(metrics)
    
    # Calculate averages and standard deviations
    avg_metrics = {}
    std_metrics = {}
    
    for metric in ['acc', 'bacc', 'f1', 'auc']:
        values = [r[metric] for r in results]
        avg_metrics[metric] = np.mean(values)
        std_metrics[metric] = np.std(values)
    
    return avg_metrics, std_metrics

# 1. Bayesian Optimization (Tree-structured Parzen Estimator)
def optimize_mlp_tpe(X_train, y_train, X_val, y_val, n_trials=50, n_runs_per_trial=15):
    """
    Optimize MLP hyperparameters using TPE.
    
    Parameters:
    - X_train, y_train: Training data
    - X_val, y_val: Validation data
    - n_trials: Number of trials for optimization
    - n_runs_per_trial: Number of runs per trial to average results
    
    Returns:
    - Best hyperparameters and results
    """
    def objective(trial):
        params = {
            'n_layers': trial.suggest_int('n_layers', 1, 5),
            'units': trial.suggest_int('units', 16, 256, log=True),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'elu']),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        }
        
        avg_metrics, _ = train_and_evaluate_multiple(
            create_mlp_model, params, X_train, y_train, X_val, y_val, n_runs=n_runs_per_trial
        )
        
        return avg_metrics['auc']
    
    study = create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    
    # Get best parameters and metrics
    best_params = study.best_params
    best_trial = study.best_trial
    
    return best_params, best_trial.value

def optimize_1d_cnn_tpe(X_train, y_train, X_val, y_val, n_trials=50, n_runs_per_trial=15):
    """
    Optimize 1D CNN hyperparameters using TPE.
    
    Parameters:
    - X_train, y_train: Training data
    - X_val, y_val: Validation data
    - n_trials: Number of trials for optimization
    - n_runs_per_trial: Number of runs per trial to average results
    
    Returns:
    - Best hyperparameters and results
    """
    def objective(trial):
        # Determine chunk length (sequence length)
        chunk_length = trial.suggest_int('chunk_length', 5, 30)
        
        # Prepare sequences
        X_train_seq, y_train_seq = prepare_sequences(X_train, y_train, chunk_length)
        X_val_seq, y_val_seq = prepare_sequences(X_val, y_val, chunk_length)
        
        params = {
            'n_conv_layers': trial.suggest_int('n_conv_layers', 1, 3),
            'filters': trial.suggest_int('filters', 16, 128, log=True),
            'kernel_size': trial.suggest_int('kernel_size', 2, 5),
            'dense_units': trial.suggest_int('dense_units', 16, 256, log=True),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'elu']),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        }
        
        avg_metrics, _ = train_and_evaluate_multiple(
            create_1d_cnn_model, params, X_train_seq, y_train_seq, X_val_seq, y_val_seq, 
            n_runs=n_runs_per_trial, is_sequence=True
        )
        
        return avg_metrics['auc']
    
    study = create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    
    # Get best parameters and metrics
    best_params = study.best_params
    best_trial = study.best_trial
    
    return best_params, best_trial.value

def optimize_lstm_tpe(X_train, y_train, X_val, y_val, n_trials=50, n_runs_per_trial=15):
    """
    Optimize LSTM hyperparameters using TPE.
    
    Parameters:
    - X_train, y_train: Training data
    - X_val, y_val: Validation data
    - n_trials: Number of trials for optimization
    - n_runs_per_trial: Number of runs per trial to average results
    
    Returns:
    - Best hyperparameters and results
    """
    def objective(trial):
        # Determine chunk length (sequence length)
        chunk_length = trial.suggest_int('chunk_length', 5, 30)
        
        # Prepare sequences
        X_train_seq, y_train_seq = prepare_sequences(X_train, y_train, chunk_length)
        X_val_seq, y_val_seq = prepare_sequences(X_val, y_val, chunk_length)
        
        params = {
            'n_lstm_layers': trial.suggest_int('n_lstm_layers', 1, 3),
            'lstm_units': trial.suggest_int('lstm_units', 16, 128, log=True),
            'dense_units': trial.suggest_int('dense_units', 16, 256, log=True),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'elu']),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        }
        
        avg_metrics, _ = train_and_evaluate_multiple(
            create_lstm_model, params, X_train_seq, y_train_seq, X_val_seq, y_val_seq, 
            n_runs=n_runs_per_trial, is_sequence=True
        )
        
        return avg_metrics['auc']
    
    study = create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    
    # Get best parameters and metrics
    best_params = study.best_params
    best_trial = study.best_trial
    
    return best_params, best_trial.value

# 2. Hyperband Method
class MLPHyperModel(kt.HyperModel):
    def __init__(self, input_dim):
        self.input_dim = input_dim
        
    def build(self, hp):
        model = Sequential()
        
        # Input layer
        model.add(Dense(
            hp.Int('units', 16, 256, step=16),
            activation=hp.Choice('activation', ['relu', 'tanh', 'elu']),
            input_shape=(self.input_dim,)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)))
        
        # Hidden layers
        for i in range(hp.Int('n_layers', 1, 5) - 1):
            model.add(Dense(
                hp.Int('units', 16, 256, step=16),
                activation=hp.Choice('activation', ['relu', 'tanh', 'elu'])
            ))
            model.add(BatchNormalization())
            model.add(Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=hp.Float('learning_rate', 1e-5, 1e-2, sampling='log')),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model

class CNNHyperModel(kt.HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        
    def build(self, hp):
        model = Sequential()
        
        # First Conv layer
        model.add(Conv1D(
            filters=hp.Int('filters', 16, 128, step=16),
            kernel_size=hp.Int('kernel_size', 2, 5),
            activation=hp.Choice('activation', ['relu', 'tanh', 'elu']),
            input_shape=self.input_shape
        ))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)))
        
        # Additional Conv layers
        for i in range(hp.Int('n_conv_layers', 1, 3) - 1):
            model.add(Conv1D(
                filters=hp.Int('filters', 16, 128, step=16),
                kernel_size=hp.Int('kernel_size', 2, 5),
                activation=hp.Choice('activation', ['relu', 'tanh', 'elu'])
            ))
            model.add(BatchNormalization())
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)))
        
        # Flatten and Dense layers
        model.add(Flatten())
        model.add(Dense(
            hp.Int('dense_units', 16, 256, step=16),
            activation=hp.Choice('activation', ['relu', 'tanh', 'elu'])
        ))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)))
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=hp.Float('learning_rate', 1e-5, 1e-2, sampling='log')),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model

class LSTMHyperModel(kt.HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        
    def build(self, hp):
        model = Sequential()
        
        # First LSTM layer
        n_lstm_layers = hp.Int('n_lstm_layers', 1, 3)
        return_sequences = n_lstm_layers > 1
        
        model.add(LSTM(
            units=hp.Int('lstm_units', 16, 128, step=16),
            return_sequences=return_sequences,
            input_shape=self.input_shape
        ))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)))
        
        # Additional LSTM layers
        for i in range(n_lstm_layers - 1):
            return_sequences = i < n_lstm_layers - 2
            model.add(LSTM(
                units=hp.Int('lstm_units', 16, 128, step=16),
                return_sequences=return_sequences
            ))
            model.add(BatchNormalization())
            model.add(Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)))
        
        # Dense layers
        model.add(Dense(
            hp.Int('dense_units', 16, 256, step=16),
            activation=hp.Choice('activation', ['relu', 'tanh', 'elu'])
        ))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout', 0.1, 0.5, step=0.1)))
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=hp.Float('learning_rate', 1e-5, 1e-2, sampling='log')),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model

def optimize_mlp_hyperband(X_train, y_train, X_val, y_val, max_epochs=80, factor=3, executions_per_trial=15):
    """
    Optimize MLP hyperparameters using Hyperband.
    
    Parameters:
    - X_train, y_train: Training data
    - X_val, y_val: Validation data
    - max_epochs: Maximum number of epochs per model
    - factor: Reduction factor for successive halving
    - executions_per_trial: Number of times to train each model configuration
    
    Returns:
    - Best hyperparameters and results
    """
    # Define the hypermodel
    hypermodel = MLPHyperModel(input_dim=X_train.shape[1])
    
    # Create the tuner
    tuner = kt.Hyperband(
        hypermodel,
        objective='val_accuracy',
        max_epochs=max_epochs,
        factor=factor,
        directory='hyperband_mlp',
        project_name='financial_ts_mlp',
        executions_per_trial=executions_per_trial,  # For random seed variation mitigation
        overwrite=True
    )
    
    # Define early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Search for the best hyperparameters
    tuner.search(
        X_train, y_train,
        epochs=max_epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Get the best hyperparameters
    best_params = tuner.get_best_hyperparameters(1)[0].values
    
    # Build the model with the best hyperparameters and evaluate it
    model = tuner.hypermodel.build(tuner.get_best_hyperparameters(1)[0])
    
    # Train and evaluate multiple times
    avg_metrics, std_metrics = train_and_evaluate_multiple(
        lambda params, input_dim: model, best_params, X_train, y_train, X_val, y_val, 
        n_runs=15  # Run 15 times with different seeds
    )
    
    return best_params, avg_metrics['auc'], model

def optimize_1d_cnn_hyperband(X_train, y_train, X_val, y_val, max_epochs=80, factor=3, executions_per_trial=15):
    """
    Optimize 1D CNN hyperparameters using Hyperband.
    
    Parameters:
    - X_train, y_train: Training data
    - X_val, y_val: Validation data
    - max_epochs: Maximum number of epochs per model
    - factor: Reduction factor for successive halving
    - executions_per_trial: Number of times to train each model configuration
    
    Returns:
    - Best hyperparameters and results
    """
    # For CNNs we need to determine the chunk_length first
    # We'll try a few different values and pick the best
    chunk_lengths = [5, 10, 15, 20, 25, 30]
    best_auc = 0
    best_chunk_length = 0
    best_model = None
    best_params = None
    
    for chunk_length in chunk_lengths:
        print(f"Trying chunk_length = {chunk_length}")
        
        # Prepare sequences
        X_train_seq, y_train_seq = prepare_sequences(X_train, y_train, chunk_length)
        X_val_seq, y_val_seq = prepare_sequences(X_val, y_val, chunk_length)
        
        # Define the hypermodel
        hypermodel = CNNHyperModel(input_shape=(chunk_length, X_train.shape[1]))
        
        # Create the tuner
        tuner = kt.Hyperband(
            hypermodel,
            objective='val_accuracy',
            max_epochs=max_epochs,
            factor=factor,
            directory=f'hyperband_cnn_{chunk_length}',
            project_name='financial_ts_cnn',
            executions_per_trial=executions_per_trial,
            overwrite=True
        )
        
        # Define early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Search for the best hyperparameters
        tuner.search(
            X_train_seq, y_train_seq,
            epochs=max_epochs,
            validation_data=(X_val_seq, y_val_seq),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Get the best hyperparameters
        params = tuner.get_best_hyperparameters(1)[0].values
        params['chunk_length'] = chunk_length
        
        # Build the model with the best hyperparameters and evaluate it
        model = tuner.hypermodel.build(tuner.get_best_hyperparameters(1)[0])
        
        # Train and evaluate multiple times
        avg_metrics, std_metrics = train_and_evaluate_multiple(
            lambda params, input_shape: model, params, X_train_seq, y_train_seq, X_val_seq, y_val_seq, 
            n_runs=15, is_sequence=True
        )
        
        # Check if this is the best model so far
        if avg_metrics['auc'] > best_auc:
            best_auc = avg_metrics['auc']
            best_chunk_length = chunk_length
            best_model = model
            best_params = params
    
    return best_params, best_auc, best_model

def optimize_lstm_hyperband(X_train, y_train, X_val, y_val, max_epochs=80, factor=3, executions_per_trial=15):
    """
    Optimize LSTM hyperparameters using Hyperband.
    
    Parameters:
    - X_train, y_train: Training data
    - X_val, y_val: Validation data
    - max_epochs: Maximum number of epochs per model
    - factor: Reduction factor for successive halving
    - executions_per_trial: Number of times to train each model configuration
    
    Returns:
    - Best hyperparameters and results
    """
    # For LSTMs we need to determine the chunk_length first
    # We'll try a few different values and pick the best
    chunk_lengths = [5, 10, 15, 20, 25, 30]
    best_auc = 0
    best_chunk_length = 0
    best_model = None
    best_params = None
    
    for chunk_length in chunk_lengths:
        print(f"Trying chunk_length = {chunk_length}")
        
        # Prepare sequences
        X_train_seq, y_train_seq = prepare_sequences(X_train, y_train, chunk_length)
        X_val_seq, y_val_seq = prepare_sequences(X_val, y_val, chunk_length)
        
        # Define the hypermodel
        hypermodel = LSTMHyperModel(input_shape=(chunk_length, X_train.shape[1]))
        
        # Create the tuner
        tuner = kt.Hyperband(
            hypermodel,
            objective='val_accuracy',
            max_epochs=max_epochs,
            factor=factor,
            directory=f'hyperband_lstm_{chunk_length}',
            project_name='financial_ts_lstm',
            executions_per_trial=executions_per_trial,
            overwrite=True
        )
        
        # Define early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Search for the best hyperparameters
        tuner.search(
            X_train_seq, y_train_seq,
            epochs=max_epochs,
            validation_data=(X_val_seq, y_val_seq),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Get the best hyperparameters
        params = tuner.get_best_hyperparameters(1)[0].values
        params['chunk_length'] = chunk_length
        
        # Build the model with the best hyperparameters and evaluate it
        model = tuner.hypermodel.build(tuner.get_best_hyperparameters(1)[0])
        
        # Train and evaluate multiple times
        avg_metrics, std_metrics = train_and_evaluate_multiple(
            lambda params, input_shape: model, params, X_train_seq, y_train_seq, X_val_seq, y_val_seq, 
            n_runs=15, is_sequence=True
        )
        
        # Check if this is the best model so far
        if avg_metrics['auc'] > best_auc:
            best_auc = avg_metrics['auc']
            best_chunk_length = chunk_length
            best_model = model
            best_params = params
    
    return best_params, best_auc, best_model

# 3. Reinforcement Learning Method (simplified implementation)
class RLController:
    """Simplified RL Controller for NAS based on the paper's algorithm."""
    
    def __init__(self, search_space, model_type='mlp'):
        self.search_space = search_space
        self.model_type = model_type
        self.trials = []
        self.best_params = None
        self.best_auc = 0
        
    def sample_params(self):
        """Sample hyperparameters from the search space."""
        params = {}
        for param_name, param_values in self.search_space.items():
            params[param_name] = np.random.choice(param_values)
        return params
    
    def update_policy(self, params, auc):
        """Update the controller's policy based on the performance."""
        # In a full implementation, this would update the RNN controller
        # For this simplified version, we just track the best parameters
        self.trials.append((params, auc))
        
        if auc > self.best_auc:
            self.best_auc = auc
            self.best_params = params
        
        # Simple exploitation: bias towards parameter values that performed well
        # This is a very simplified approach compared to the paper's RNN controller
        for param_name, param_values in self.search_space.items():
            # Sort param values by average AUC
            param_aucs = {}
            for value in param_values:
                relevant_trials = [(p, a) for p, a in self.trials if p[param_name] == value]
                if relevant_trials:
                    param_aucs[value] = np.mean([a for _, a in relevant_trials])
                else:
                    param_aucs[value] = 0
            
            # Update probabilities for this parameter (not actually used in the simplified version)
            # In a full implementation, these would adjust the RNN controller's weights
    
    def search(self, X_train, y_train, X_val, y_val, n_trials=50, n_runs_per_trial=15):
        """
        Search for the best hyperparameters using a simplified RL approach.
        
        Parameters:
        - X_train, y_train: Training data
        - X_val, y_val: Validation data
        - n_trials: Number of trials
        - n_runs_per_trial: Number of runs per trial to average results
        
        Returns:
        - Best hyperparameters and results
        """
        for i in range(n_trials):
            print(f"RL Trial {i+1}/{n_trials}")
            
            # Sample hyperparameters
            params = self.sample_params()
            
            # For sequence models, prepare the data
            if self.model_type in ['cnn', 'lstm']:
                chunk_length = params['chunk_length']
                X_train_seq, y_train_seq = prepare_sequences(X_train, y_train, chunk_length)
                X_val_seq, y_val_seq = prepare_sequences(X_val, y_val, chunk_length)
                
                # Train and evaluate the model
                if self.model_type == 'cnn':
                    avg_metrics, _ = train_and_evaluate_multiple(
                        create_1d_cnn_model, params, X_train_seq, y_train_seq, X_val_seq, y_val_seq, 
                        n_runs=n_runs_per_trial, is_sequence=True
                    )
                else:  # lstm
                    avg_metrics, _ = train_and_evaluate_multiple(
                        create_lstm_model, params, X_train_seq, y_train_seq, X_val_seq, y_val_seq, 
                        n_runs=n_runs_per_trial, is_sequence=True
                    )
            else:  # mlp
                avg_metrics, _ = train_and_evaluate_multiple(
                    create_mlp_model, params, X_train, y_train, X_val, y_val, 
                    n_runs=n_runs_per_trial
                )
            
            # Update policy based on the performance
            self.update_policy(params, avg_metrics['auc'])
            
            print(f"  Trial AUC: {avg_metrics['auc']:.4f}, Best AUC so far: {self.best_auc:.4f}")
        
        return self.best_params, self.best_auc

def optimize_mlp_rl(X_train, y_train, X_val, y_val, n_trials=50, n_runs_per_trial=15):
    """
    Optimize MLP hyperparameters using a simplified RL approach.
    
    Parameters:
    - X_train, y_train: Training data
    - X_val, y_val: Validation data
    - n_trials: Number of trials
    - n_runs_per_trial: Number of runs per trial to average results
    
    Returns:
    - Best hyperparameters and results
    """
    search_space = {
        'n_layers': [1, 2, 3, 4, 5],
        'units': [16, 32, 64, 128, 256],
        'activation': ['relu', 'tanh', 'elu'],
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
        'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    }
    
    controller = RLController(search_space, model_type='mlp')
    best_params, best_auc = controller.search(X_train, y_train, X_val, y_val, n_trials, n_runs_per_trial)
    
    return best_params, best_auc

def optimize_1d_cnn_rl(X_train, y_train, X_val, y_val, n_trials=50, n_runs_per_trial=15):
    """
    Optimize 1D CNN hyperparameters using a simplified RL approach.
    
    Parameters:
    - X_train, y_train: Training data
    - X_val, y_val: Validation data
    - n_trials: Number of trials
    - n_runs_per_trial: Number of runs per trial to average results
    
    Returns:
    - Best hyperparameters and results
    """
    search_space = {
        'chunk_length': [5, 10, 15, 20, 25, 30],
        'n_conv_layers': [1, 2, 3],
        'filters': [16, 32, 64, 128],
        'kernel_size': [2, 3, 4, 5],
        'dense_units': [16, 32, 64, 128, 256],
        'activation': ['relu', 'tanh', 'elu'],
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
        'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    }
    
    controller = RLController(search_space, model_type='cnn')
    best_params, best_auc = controller.search(X_train, y_train, X_val, y_val, n_trials, n_runs_per_trial)
    
    return best_params, best_auc

def optimize_lstm_rl(X_train, y_train, X_val, y_val, n_trials=50, n_runs_per_trial=15):
    """
    Optimize LSTM hyperparameters using a simplified RL approach.
    
    Parameters:
    - X_train, y_train: Training data
    - X_val, y_val: Validation data
    - n_trials: Number of trials
    - n_runs_per_trial: Number of runs per trial to average results
    
    Returns:
    - Best hyperparameters and results
    """
    search_space = {
        'chunk_length': [5, 10, 15, 20, 25, 30],
        'n_lstm_layers': [1, 2, 3],
        'lstm_units': [16, 32, 64, 128],
        'dense_units': [16, 32, 64, 128, 256],
        'activation': ['relu', 'tanh', 'elu'],
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
        'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    }
    
    controller = RLController(search_space, model_type='lstm')
    best_params, best_auc = controller.search(X_train, y_train, X_val, y_val, n_trials, n_runs_per_trial)
    
    return best_params, best_auc

# Function to compare all methods
def compare_nas_methods(X_train, y_train, X_val, y_val, X_test, y_test, n_trials=50, n_runs_per_trial=15, final_runs=50):
    """
    Compare all NAS methods on the same dataset.
    
    Parameters:
    - X_train, y_train: Training data
    - X_val, y_val: Validation data
    - X_test, y_test: Test data
    - n_trials: Number of trials for each method
    - n_runs_per_trial: Number of runs per trial to average results
    - final_runs: Number of runs for final evaluation
    
    Returns:
    - Results dictionary
    """
    results = {}
    
    # 1. Bayesian Optimization (TPE)
    print("\n=== Bayesian Optimization (TPE) ===")
    
    print("\nOptimizing MLP...")
    mlp_tpe_params, mlp_tpe_val_auc = optimize_mlp_tpe(X_train, y_train, X_val, y_val, n_trials, n_runs_per_trial)
    
    print("\nOptimizing 1D CNN...")
    cnn_tpe_params, cnn_tpe_val_auc = optimize_1d_cnn_tpe(X_train, y_train, X_val, y_val, n_trials, n_runs_per_trial)
    
    print("\nOptimizing LSTM...")
    lstm_tpe_params, lstm_tpe_val_auc = optimize_lstm_tpe(X_train, y_train, X_val, y_val, n_trials, n_runs_per_trial)
    
    # 2. Hyperband
    print("\n=== Hyperband Method ===")
    
    print("\nOptimizing MLP...")
    mlp_hb_params, mlp_hb_val_auc, _ = optimize_mlp_hyperband(X_train, y_train, X_val, y_val, 80, 3, n_runs_per_trial)
    
    print("\nOptimizing 1D CNN...")
    cnn_hb_params, cnn_hb_val_auc, _ = optimize_1d_cnn_hyperband(X_train, y_train, X_val, y_val, 80, 3, n_runs_per_trial)
    
    print("\nOptimizing LSTM...")
    lstm_hb_params, lstm_hb_val_auc, _ = optimize_lstm_hyperband(X_train, y_train, X_val, y_val, 80, 3, n_runs_per_trial)
    
    # 3. Reinforcement Learning
    print("\n=== Reinforcement Learning Method ===")
    
    print("\nOptimizing MLP...")
    mlp_rl_params, mlp_rl_val_auc = optimize_mlp_rl(X_train, y_train, X_val, y_val, n_trials, n_runs_per_trial)
    
    print("\nOptimizing 1D CNN...")
    cnn_rl_params, cnn_rl_val_auc = optimize_1d_cnn_rl(X_train, y_train, X_val, y_val, n_trials, n_runs_per_trial)
    
    print("\nOptimizing LSTM...")
    lstm_rl_params, lstm_rl_val_auc = optimize_lstm_rl(X_train, y_train, X_val, y_val, n_trials, n_runs_per_trial)
    
    # Final evaluation on test set
    print("\n=== Final Evaluation on Test Set ===")
    
    # Prepare models for final evaluation
    models_to_evaluate = {
        'MLP_TPE': (create_mlp_model, mlp_tpe_params, X_train, y_train, X_test, y_test, False),
        'MLP_HB': (create_mlp_model, mlp_hb_params, X_train, y_train, X_test, y_test, False),
        'MLP_RL': (create_mlp_model, mlp_rl_params, X_train, y_train, X_test, y_test, False),
        
        'CNN_TPE': (create_1d_cnn_model, cnn_tpe_params, 
                  *prepare_sequences(X_train, y_train, cnn_tpe_params['chunk_length']),
                  *prepare_sequences(X_test, y_test, cnn_tpe_params['chunk_length']), True),
        'CNN_HB': (create_1d_cnn_model, cnn_hb_params, 
                 *prepare_sequences(X_train, y_train, cnn_hb_params['chunk_length']),
                 *prepare_sequences(X_test, y_test, cnn_hb_params['chunk_length']), True),
        'CNN_RL': (create_1d_cnn_model, cnn_rl_params, 
                 *prepare_sequences(X_train, y_train, cnn_rl_params['chunk_length']),
                 *prepare_sequences(X_test, y_test, cnn_rl_params['chunk_length']), True),
        
        'LSTM_TPE': (create_lstm_model, lstm_tpe_params, 
                   *prepare_sequences(X_train, y_train, lstm_tpe_params['chunk_length']),
                   *prepare_sequences(X_test, y_test, lstm_tpe_params['chunk_length']), True),
        'LSTM_HB': (create_lstm_model, lstm_hb_params, 
                  *prepare_sequences(X_train, y_train, lstm_hb_params['chunk_length']),
                  *prepare_sequences(X_test, y_test, lstm_hb_params['chunk_length']), True),
        'LSTM_RL': (create_lstm_model, lstm_rl_params, 
                  *prepare_sequences(X_train, y_train, lstm_rl_params['chunk_length']),
                  *prepare_sequences(X_test, y_test, lstm_rl_params['chunk_length']), True)
    }
    
    # Evaluate each model on the test set
    final_results = {}
    
    for name, (model_func, params, X_train_eval, y_train_eval, X_test_eval, y_test_eval, is_sequence) in models_to_evaluate.items():
        print(f"\nEvaluating {name}...")
        
        avg_metrics, std_metrics = train_and_evaluate_multiple(
            model_func, params, X_train_eval, y_train_eval, X_test_eval, y_test_eval, 
            n_runs=final_runs, is_sequence=is_sequence
        )
        
        final_results[name] = {
            'params': params,
            'val_auc': locals()[f"{name.lower()}_val_auc"],
            'test_metrics': avg_metrics,
            'test_std': std_metrics
        }
        
        print(f"  Test AUC: {avg_metrics['auc']:.4f} ± {std_metrics['auc']:.4f}")
        print(f"  Test Balanced Accuracy: {avg_metrics['bacc']:.4f} ± {std_metrics['bacc']:.4f}")
        print(f"  Test F1: {avg_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")
    
    # Plot results
    plot_comparison_results(final_results)
    
    return final_results

def plot_comparison_results(results):
    """
    Plot comparison of different NAS methods.
    
    Parameters:
    - results: Dictionary of results
    """
    # Extract data for plotting
    methods = ['TPE', 'HB', 'RL']
    architectures = ['MLP', 'CNN', 'LSTM']
    
    metrics = ['auc', 'bacc', 'f1']
    metric_labels = ['AUC', 'Balanced Accuracy', 'F1 Score']
    
    # Create figure
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 15))
    
    for i, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        # Prepare data for this metric
        data = []
        errors = []
        labels = []
        
        for arch in architectures:
            for method in methods:
                key = f"{arch}_{method}"
                if key in results:
                    data.append(results[key]['test_metrics'][metric])
                    errors.append(results[key]['test_std'][metric])
                    labels.append(key)
        
        # Plot
        x = np.arange(len(labels))
        bars = axes[i].bar(x, data, yerr=errors, capsize=5)
        
        # Color bars by architecture
        colors = {'MLP': 'skyblue', 'CNN': 'lightgreen', 'LSTM': 'salmon'}
        for j, bar in enumerate(bars):
            arch = labels[j].split('_')[0]
            bar.set_color(colors[arch])
        
        # Add labels and formatting
        axes[i].set_ylabel(metric_label)
        axes[i].set_title(f'Test {metric_label} Comparison')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(labels, rotation=45, ha='right')
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for j, v in enumerate(data):
            axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Add legend
    handles = [plt.Rectangle((0,0),1,1, color=colors[arch]) for arch in architectures]
    fig.legend(handles, architectures, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('nas_comparison_results.png')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Generate simulated data
    print("Generating simulated financial data...")
    data, target = generate_simulated_data(n_samples=4000, n_features=100)
    
    print(f"Generated data shape: {data.shape}, target shape: {target.shape}")
    
    # Preprocess the data
    print("\nPreprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
        data, target, remove_time_derived=True, apply_pca=True, n_components=50
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Compare NAS methods (reduced number of trials for demonstration)
    results = compare_nas_methods(
        X_train, y_train, X_val, y_val, X_test, y_test, 
        n_trials=10,  # Reduced for demonstration
        n_runs_per_trial=5,  # Reduced for demonstration
        final_runs=10  # Reduced for demonstration
    )