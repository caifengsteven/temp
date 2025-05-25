"""
Improved Deep Hedging Model Implementation

This module implements an improved deep hedging model using TensorFlow.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Dropout, Concatenate, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
import os

class ImprovedDeepHedgingModel:
    """
    Improved Deep Hedging Model
    
    This class implements an improved deep hedging model using TensorFlow.
    The model takes market features as input and outputs the optimal hedge ratio.
    """
    
    def __init__(self, lookback_period=20, feature_dim=8, lstm_units=128, dense_units=64,
                 learning_rate=0.0005, lambda_reg=0.005, risk_aversion=0.5, use_gru=True,
                 bidirectional=True, dropout_rate=0.3):
        """
        Initialize the improved deep hedging model
        
        Parameters:
        -----------
        lookback_period : int
            Number of time steps to look back
        feature_dim : int
            Number of features per time step
        lstm_units : int
            Number of LSTM/GRU units
        dense_units : int
            Number of dense units
        learning_rate : float
            Learning rate for the optimizer
        lambda_reg : float
            Regularization parameter
        risk_aversion : float
            Risk aversion parameter for the utility function
        use_gru : bool
            Whether to use GRU instead of LSTM
        bidirectional : bool
            Whether to use bidirectional RNNs
        dropout_rate : float
            Dropout rate for regularization
        """
        self.lookback_period = lookback_period
        self.feature_dim = feature_dim
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.risk_aversion = risk_aversion
        self.use_gru = use_gru
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate
        
        # Initialize scalers
        self.feature_scaler = None
        self.target_scaler = None
        
        # Build the model
        self.model = self._build_model()
    
    def _build_model(self):
        """Build the improved deep hedging model"""
        # Input layers
        market_features = Input(shape=(self.lookback_period, self.feature_dim), name='market_features')
        
        # RNN layers (LSTM or GRU)
        if self.use_gru:
            rnn_layer = GRU
        else:
            rnn_layer = LSTM
        
        # First RNN layer
        if self.bidirectional:
            rnn_out = Bidirectional(rnn_layer(self.lstm_units, return_sequences=True))(market_features)
        else:
            rnn_out = rnn_layer(self.lstm_units, return_sequences=True)(market_features)
        
        # Batch normalization and dropout
        rnn_out = BatchNormalization()(rnn_out)
        rnn_out = Dropout(self.dropout_rate)(rnn_out)
        
        # Second RNN layer
        if self.bidirectional:
            rnn_out = Bidirectional(rnn_layer(self.lstm_units // 2, return_sequences=False))(rnn_out)
        else:
            rnn_out = rnn_layer(self.lstm_units // 2, return_sequences=False)(rnn_out)
        
        # Batch normalization and dropout
        rnn_out = BatchNormalization()(rnn_out)
        rnn_out = Dropout(self.dropout_rate)(rnn_out)
        
        # Dense layers
        dense_out = Dense(self.dense_units, activation='relu')(rnn_out)
        dense_out = BatchNormalization()(dense_out)
        dense_out = Dropout(self.dropout_rate)(dense_out)
        
        dense_out = Dense(self.dense_units // 2, activation='relu')(dense_out)
        dense_out = BatchNormalization()(dense_out)
        dense_out = Dropout(self.dropout_rate / 2)(dense_out)
        
        # Output layer (hedge ratio)
        # Using sigmoid activation to constrain hedge ratio between 0 and 1, then scaling to [-1, 1]
        hedge_ratio_raw = Dense(1, activation='sigmoid', name='hedge_ratio_raw')(dense_out)
        hedge_ratio = tf.keras.layers.Lambda(lambda x: 2 * x - 1, name='hedge_ratio')(hedge_ratio_raw)
        
        # Create model
        model = Model(inputs=market_features, outputs=hedge_ratio)
        
        # Compile model with improved mean-variance utility loss function
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=self._improved_mean_variance_utility_loss
        )
        
        return model
    
    def _improved_mean_variance_utility_loss(self, y_true, y_pred):
        """
        Improved mean-variance utility loss function
        
        U(X) = E[X] - λ * Var[X] - α * CVaR(X)
        
        where X is the P&L of the hedged portfolio
        """
        # In this improved version, y_true represents the returns of the asset to be hedged
        # y_pred represents the hedge ratio
        
        # Calculate hedged portfolio returns
        # We assume the hedge instrument return is negatively correlated with the asset
        # but with a more realistic correlation
        hedge_return = -0.8 * y_true + 0.2 * tf.random.normal(tf.shape(y_true), mean=0.0, stddev=0.01)
        hedged_return = y_true + y_pred * hedge_return
        
        # Calculate mean and variance
        mean_return = tf.reduce_mean(hedged_return)
        # Calculate variance manually
        var_return = tf.reduce_mean(tf.square(hedged_return - mean_return))
        
        # Calculate Conditional Value at Risk (CVaR)
        # First, sort the returns
        sorted_returns = tf.sort(hedged_return, axis=0)
        # Calculate the 5% worst returns (CVaR at 95% confidence level)
        n = tf.shape(sorted_returns)[0]
        cvar_cutoff = tf.cast(tf.math.ceil(0.05 * tf.cast(n, tf.float32)), tf.int32)
        worst_returns = sorted_returns[:cvar_cutoff]
        cvar = tf.reduce_mean(worst_returns)
        
        # Mean-variance-CVaR utility (negative because we want to maximize utility)
        # Adding a small constant to avoid numerical issues
        utility = mean_return - self.risk_aversion * var_return - 0.5 * tf.abs(cvar)
        
        # Add regularization term with L1 and L2 components (elastic net)
        reg_term = self.lambda_reg * (
            0.5 * tf.reduce_mean(tf.square(y_pred)) +  # L2 regularization
            0.5 * tf.reduce_mean(tf.abs(y_pred))       # L1 regularization
        )
        
        # Add a term to penalize large changes in hedge ratio
        if tf.shape(y_pred)[0] > 1:
            hedge_ratio_changes = y_pred[1:] - y_pred[:-1]
            smoothness_penalty = 0.01 * tf.reduce_mean(tf.square(hedge_ratio_changes))
        else:
            smoothness_penalty = 0.0
        
        # Return negative utility (to minimize)
        return -utility + reg_term + smoothness_penalty
    
    def prepare_data(self, data, target_col, feature_cols, train_ratio=0.8, use_robust_scaling=True):
        """
        Prepare data for the deep hedging model with improved scaling
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data
        target_col : str
            Target column name (asset returns)
        feature_cols : list
            List of feature column names
        train_ratio : float
            Ratio of data to use for training
        use_robust_scaling : bool
            Whether to use robust scaling (less sensitive to outliers)
            
        Returns:
        --------
        tuple
            (X_train, y_train, X_test, y_test, dates_train, dates_test)
        """
        # Ensure all data is numeric
        for col in feature_cols + [target_col]:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Drop rows with NaN values
        data = data.dropna()
        
        # Initialize scalers
        if use_robust_scaling:
            self.feature_scaler = RobustScaler()
            self.target_scaler = RobustScaler()
        else:
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        
        # Extract features and target
        features = data[feature_cols].values
        target = data[target_col].values.reshape(-1, 1)
        dates = data.index
        
        # Fit and transform the data
        scaled_features = self.feature_scaler.fit_transform(features)
        scaled_target = self.target_scaler.fit_transform(target)
        
        # Reshape the scaled data back to DataFrame
        scaled_features_df = pd.DataFrame(
            scaled_features, 
            index=data.index, 
            columns=feature_cols
        )
        
        scaled_target_series = pd.Series(
            scaled_target.flatten(), 
            index=data.index
        )
        
        # Create sequences
        X, y, sequence_dates = [], [], []
        for i in range(len(scaled_features_df) - self.lookback_period):
            X.append(scaled_features_df.iloc[i:i+self.lookback_period].values)
            y.append(scaled_target_series.iloc[i+self.lookback_period])
            sequence_dates.append(dates[i+self.lookback_period])
        
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # Split into train and test sets
        train_size = int(len(X) * train_ratio)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        dates_train = sequence_dates[:train_size]
        dates_test = sequence_dates[train_size:]
        
        return X_train, y_train, X_test, y_test, dates_train, dates_test
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, 
              patience=20, min_delta=0.0001, verbose=1):
        """
        Train the deep hedging model with early stopping and learning rate reduction
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training targets
        X_val : numpy.ndarray
            Validation features
        y_val : numpy.ndarray
            Validation targets
        epochs : int
            Maximum number of epochs
        batch_size : int
            Batch size
        patience : int
            Patience for early stopping
        min_delta : float
            Minimum change in loss for early stopping
        verbose : int
            Verbosity level
            
        Returns:
        --------
        tensorflow.keras.callbacks.History
            Training history
        """
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=patience,
                min_delta=min_delta,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train the model
        if X_val is not None and y_val is not None:
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=verbose
            )
        else:
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=verbose
            )
        
        return history
    
    def predict(self, X):
        """
        Predict hedge ratios
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input features
            
        Returns:
        --------
        numpy.ndarray
            Predicted hedge ratios
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model
        
        Parameters:
        -----------
        X_test : numpy.ndarray
            Test features
        y_test : numpy.ndarray
            Test targets
            
        Returns:
        --------
        float
            Loss value
        """
        return self.model.evaluate(X_test, y_test)
    
    def save_model(self, filepath):
        """Save the model and scalers"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save the model
        self.model.save(filepath)
        
        # Save scalers if they exist
        if self.feature_scaler is not None and self.target_scaler is not None:
            import joblib
            scaler_path = filepath.replace('.h5', '_scalers.joblib')
            joblib.dump({
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scaler
            }, scaler_path)
    
    def load_model(self, filepath):
        """Load the model and scalers"""
        # Load the model
        self.model = tf.keras.models.load_model(
            filepath,
            custom_objects={'_improved_mean_variance_utility_loss': self._improved_mean_variance_utility_loss}
        )
        
        # Load scalers if they exist
        scaler_path = filepath.replace('.h5', '_scalers.joblib')
        if os.path.exists(scaler_path):
            import joblib
            scalers = joblib.load(scaler_path)
            self.feature_scaler = scalers['feature_scaler']
            self.target_scaler = scalers['target_scaler']
    
    def plot_training_history(self, history):
        """
        Plot training history
        
        Parameters:
        -----------
        history : tensorflow.keras.callbacks.History
            Training history
        """
        plt.figure(figsize=(15, 6))
        
        # Plot training & validation loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'])
            plt.legend(['Train', 'Validation'], loc='upper right')
        else:
            plt.legend(['Train'], loc='upper right')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.grid(True)
        
        # Plot learning rate if available
        if 'lr' in history.history:
            plt.subplot(1, 2, 2)
            plt.plot(history.history['lr'])
            plt.title('Learning Rate')
            plt.ylabel('Learning Rate')
            plt.xlabel('Epoch')
            plt.yscale('log')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('improved_deep_hedging_training_history.png')
        plt.close()
    
    def plot_hedge_ratios(self, X_test, dates=None, actual_returns=None):
        """
        Plot predicted hedge ratios with additional analysis
        
        Parameters:
        -----------
        X_test : numpy.ndarray
            Test features
        dates : numpy.ndarray or pandas.DatetimeIndex
            Dates corresponding to X_test
        actual_returns : numpy.ndarray
            Actual returns of the asset
        """
        # Predict hedge ratios
        hedge_ratios = self.predict(X_test)
        
        plt.figure(figsize=(15, 10))
        
        # Plot hedge ratios
        plt.subplot(2, 1, 1)
        if dates is not None:
            plt.plot(dates, hedge_ratios)
            plt.xlabel('Date')
        else:
            plt.plot(hedge_ratios)
            plt.xlabel('Time')
        
        plt.title('Predicted Hedge Ratios')
        plt.ylabel('Hedge Ratio')
        plt.grid(True)
        
        # Plot histogram of hedge ratios
        plt.subplot(2, 2, 3)
        plt.hist(hedge_ratios, bins=30)
        plt.title('Distribution of Hedge Ratios')
        plt.xlabel('Hedge Ratio')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        # Plot hedge ratio vs returns if available
        if actual_returns is not None:
            plt.subplot(2, 2, 4)
            plt.scatter(actual_returns, hedge_ratios, alpha=0.5)
            plt.title('Hedge Ratio vs Asset Returns')
            plt.xlabel('Asset Return')
            plt.ylabel('Hedge Ratio')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('improved_deep_hedging_predicted_ratios.png')
        plt.close()

def create_enhanced_synthetic_data(n_samples=2000, n_features=8, lookback=20, seed=42):
    """
    Create enhanced synthetic data for testing the deep hedging model
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features
    lookback : int
        Lookback period
    seed : int
        Random seed
        
    Returns:
    --------
    tuple
        (X_train, y_train, X_test, y_test, dates_train, dates_test)
    """
    np.random.seed(seed)
    
    # Create dates
    start_date = pd.Timestamp('2020-01-01')
    dates = pd.date_range(start=start_date, periods=n_samples, freq='B')
    
    # Generate more realistic asset returns with volatility clustering
    # Use GARCH-like process
    asset_returns = np.zeros(n_samples)
    volatility = np.zeros(n_samples)
    
    # Initial volatility
    volatility[0] = 0.01
    
    # GARCH parameters
    omega = 0.00001
    alpha = 0.1
    beta = 0.8
    
    # Generate returns with volatility clustering
    for t in range(1, n_samples):
        # Update volatility
        volatility[t] = np.sqrt(omega + alpha * asset_returns[t-1]**2 + beta * volatility[t-1]**2)
        # Generate return
        asset_returns[t] = np.random.normal(0.0005, volatility[t])
    
    # Generate correlated features
    features = np.zeros((n_samples, n_features))
    
    # Feature 1: Asset returns
    features[:, 0] = asset_returns
    
    # Feature 2: Lagged asset returns
    features[1:, 1] = asset_returns[:-1]
    features[0, 1] = 0
    
    # Feature 3: Rolling volatility (20-day)
    vol_window = 20
    for t in range(vol_window, n_samples):
        features[t, 2] = np.std(asset_returns[t-vol_window:t]) * np.sqrt(252)
    features[:vol_window, 2] = features[vol_window, 2]
    
    # Feature 4: Rolling mean (20-day)
    for t in range(vol_window, n_samples):
        features[t, 3] = np.mean(asset_returns[t-vol_window:t]) * 252
    features[:vol_window, 3] = features[vol_window, 3]
    
    # Feature 5: Momentum (sign of 20-day return)
    for t in range(vol_window, n_samples):
        features[t, 4] = np.sign(np.sum(asset_returns[t-vol_window:t]))
    features[:vol_window, 4] = features[vol_window, 4]
    
    # Feature 6: Volatility ratio (short-term / long-term)
    short_window = 5
    long_window = 20
    for t in range(long_window, n_samples):
        short_vol = np.std(asset_returns[t-short_window:t])
        long_vol = np.std(asset_returns[t-long_window:t])
        features[t, 5] = short_vol / long_vol if long_vol > 0 else 1.0
    features[:long_window, 5] = features[long_window, 5]
    
    # Feature 7: Mean reversion signal
    for t in range(vol_window, n_samples):
        mean = np.mean(asset_returns[t-vol_window:t])
        std = np.std(asset_returns[t-vol_window:t])
        if std > 0:
            features[t, 6] = (asset_returns[t] - mean) / std
        else:
            features[t, 6] = 0
    features[:vol_window, 6] = features[vol_window, 6]
    
    # Feature 8: Simulated trading volume (correlated with volatility)
    base_volume = 1000000
    features[:, 7] = base_volume * (1 + 5 * volatility + 0.2 * np.random.randn(n_samples))
    
    # Create DataFrame
    df = pd.DataFrame(
        features, 
        index=dates,
        columns=[
            'asset_return', 'lagged_return', 'rolling_vol', 'rolling_mean',
            'momentum', 'vol_ratio', 'mean_reversion', 'volume'
        ]
    )
    
    # Create model instance for data preparation
    model = ImprovedDeepHedgingModel(lookback_period=lookback, feature_dim=n_features)
    
    # Prepare data
    X_train, y_train, X_test, y_test, dates_train, dates_test = model.prepare_data(
        data=df,
        target_col='asset_return',
        feature_cols=df.columns.tolist(),
        train_ratio=0.8
    )
    
    return X_train, y_train, X_test, y_test, dates_train, dates_test

def main():
    """Main function to test the improved deep hedging model"""
    print("Testing improved deep hedging model with enhanced synthetic data...")
    
    # Create enhanced synthetic data
    X_train, y_train, X_test, y_test, dates_train, dates_test = create_enhanced_synthetic_data(
        n_samples=2000, 
        n_features=8, 
        lookback=20
    )
    
    # Initialize improved deep hedging model
    model = ImprovedDeepHedgingModel(
        lookback_period=20,
        feature_dim=8,
        lstm_units=128,
        dense_units=64,
        learning_rate=0.0005,
        lambda_reg=0.005,
        risk_aversion=0.5,
        use_gru=True,
        bidirectional=True,
        dropout_rate=0.3
    )
    
    # Train the model
    print("\nTraining the model...")
    history = model.train(
        X_train, y_train,
        X_val=X_test[:100], y_val=y_test[:100],
        epochs=100,
        batch_size=32,
        patience=20,
        verbose=1
    )
    
    # Plot training history
    model.plot_training_history(history)
    
    # Evaluate the model
    print("\nEvaluating the model...")
    loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    
    # Plot predicted hedge ratios
    model.plot_hedge_ratios(X_test, dates=dates_test, actual_returns=y_test)
    
    # Save the model
    model.save_model('improved_deep_hedging_model.h5')
    print("\nModel saved to 'improved_deep_hedging_model.h5'")

if __name__ == "__main__":
    main()
