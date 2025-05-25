"""
Deep Hedging Model Implementation

This module implements a deep hedging model using TensorFlow.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class DeepHedgingModel:
    """
    Deep Hedging Model

    This class implements a deep hedging model using TensorFlow.
    The model takes market features as input and outputs the optimal hedge ratio.
    """

    def __init__(self, lookback_period=10, feature_dim=5, lstm_units=64, dense_units=32,
                 learning_rate=0.001, lambda_reg=0.01, risk_aversion=1.0):
        """
        Initialize the deep hedging model

        Parameters:
        -----------
        lookback_period : int
            Number of time steps to look back
        feature_dim : int
            Number of features per time step
        lstm_units : int
            Number of LSTM units
        dense_units : int
            Number of dense units
        learning_rate : float
            Learning rate for the optimizer
        lambda_reg : float
            Regularization parameter
        risk_aversion : float
            Risk aversion parameter for the utility function
        """
        self.lookback_period = lookback_period
        self.feature_dim = feature_dim
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.risk_aversion = risk_aversion

        # Build the model
        self.model = self._build_model()

    def _build_model(self):
        """Build the deep hedging model"""
        # Input layers
        market_features = Input(shape=(self.lookback_period, self.feature_dim), name='market_features')

        # LSTM layers
        lstm_out = LSTM(self.lstm_units, return_sequences=True)(market_features)
        lstm_out = Dropout(0.2)(lstm_out)
        lstm_out = LSTM(self.lstm_units)(lstm_out)
        lstm_out = Dropout(0.2)(lstm_out)

        # Dense layers
        dense_out = Dense(self.dense_units, activation='relu')(lstm_out)

        # Output layer (hedge ratio)
        hedge_ratio = Dense(1, activation='tanh', name='hedge_ratio')(dense_out)

        # Create model
        model = Model(inputs=market_features, outputs=hedge_ratio)

        # Compile model with mean-variance utility loss function
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=self._mean_variance_utility_loss
        )

        return model

    def _mean_variance_utility_loss(self, y_true, y_pred):
        """
        Mean-variance utility loss function

        U(X) = E[X] - λ * Var[X]

        where X is the P&L of the hedged portfolio
        """
        # In this simplified version, y_true represents the returns of the asset to be hedged
        # y_pred represents the hedge ratio

        # Calculate hedged portfolio returns
        # Assuming a simple model where:
        # - y_true[:, 0] is the asset return
        # - We're using a single hedging instrument with return that's negatively correlated with the asset
        hedge_return = -y_true  # Simplified assumption
        hedged_return = y_true + y_pred * hedge_return

        # Calculate mean and variance
        mean_return = tf.reduce_mean(hedged_return)
        # Calculate variance manually
        var_return = tf.reduce_mean(tf.square(hedged_return - mean_return))

        # Mean-variance utility (negative because we want to maximize utility)
        utility = mean_return - self.risk_aversion * var_return

        # Add regularization term
        reg_term = self.lambda_reg * tf.reduce_mean(tf.square(y_pred))

        # Return negative utility (to minimize)
        return -utility + reg_term

    def prepare_data(self, data, target_col, feature_cols, train_ratio=0.8):
        """
        Prepare data for the deep hedging model

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

        Returns:
        --------
        tuple
            (X_train, y_train, X_test, y_test)
        """
        # Ensure all data is numeric
        for col in feature_cols + [target_col]:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Drop rows with NaN values
        data = data.dropna()

        # Normalize data
        data_norm = data.copy()
        for col in feature_cols + [target_col]:
            mean = data[col].mean()
            std = data[col].std()
            data_norm[col] = (data[col] - mean) / std

        # Create sequences
        X, y = [], []
        for i in range(len(data_norm) - self.lookback_period):
            X.append(data_norm[feature_cols].iloc[i:i+self.lookback_period].values)
            y.append(data_norm[target_col].iloc[i+self.lookback_period])

        X = np.array(X)
        y = np.array(y).reshape(-1, 1)  # Reshape to match model output

        # Split into train and test sets
        train_size = int(len(X) * train_ratio)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        return X_train, y_train, X_test, y_test

    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1):
        """
        Train the deep hedging model

        Parameters:
        -----------
        X_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training targets
        epochs : int
            Number of epochs
        batch_size : int
            Batch size
        validation_split : float
            Validation split ratio
        verbose : int
            Verbosity level

        Returns:
        --------
        tensorflow.keras.callbacks.History
            Training history
        """
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
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
        """Save the model"""
        self.model.save(filepath)

    def load_model(self, filepath):
        """Load the model"""
        self.model = tf.keras.models.load_model(
            filepath,
            custom_objects={'_mean_variance_utility_loss': self._mean_variance_utility_loss}
        )

    def plot_training_history(self, history):
        """
        Plot training history

        Parameters:
        -----------
        history : tensorflow.keras.callbacks.History
            Training history
        """
        plt.figure(figsize=(12, 4))

        # Plot training & validation loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')

        plt.tight_layout()
        plt.savefig('deep_hedging_training_history.png')
        plt.close()

    def plot_hedge_ratios(self, X_test, dates=None):
        """
        Plot predicted hedge ratios

        Parameters:
        -----------
        X_test : numpy.ndarray
            Test features
        dates : numpy.ndarray or pandas.DatetimeIndex
            Dates corresponding to X_test
        """
        # Predict hedge ratios
        hedge_ratios = self.predict(X_test)

        plt.figure(figsize=(12, 6))

        if dates is not None:
            plt.plot(dates, hedge_ratios)
            plt.xlabel('Date')
        else:
            plt.plot(hedge_ratios)
            plt.xlabel('Time')

        plt.title('Predicted Hedge Ratios')
        plt.ylabel('Hedge Ratio')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('deep_hedging_predicted_ratios.png')
        plt.close()

def create_synthetic_data(n_samples=1000, n_features=5, lookback=10, seed=42):
    """
    Create synthetic data for testing the deep hedging model

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
        (X_train, y_train, X_test, y_test)
    """
    np.random.seed(seed)

    # Generate random features
    features = np.random.normal(0, 1, (n_samples, n_features))

    # Generate target (asset returns)
    # Assume the first feature is highly correlated with the asset return
    asset_returns = 0.7 * features[:, 0] + 0.3 * np.random.normal(0, 1, n_samples)

    # Create sequences
    X, y = [], []
    for i in range(n_samples - lookback):
        X.append(features[i:i+lookback])
        y.append(asset_returns[i+lookback])

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    # Split into train and test sets (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, y_train, X_test, y_test

def main():
    """Main function to test the deep hedging model"""
    print("Testing deep hedging model with synthetic data...")

    # Create synthetic data
    X_train, y_train, X_test, y_test = create_synthetic_data(n_samples=1000, n_features=5, lookback=10)

    # Initialize deep hedging model
    model = DeepHedgingModel(
        lookback_period=10,
        feature_dim=5,
        lstm_units=64,
        dense_units=32,
        learning_rate=0.001,
        lambda_reg=0.01,
        risk_aversion=1.0
    )

    # Train the model
    print("\nTraining the model...")
    history = model.train(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # Plot training history
    model.plot_training_history(history)

    # Evaluate the model
    print("\nEvaluating the model...")
    loss = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")

    # Plot predicted hedge ratios
    model.plot_hedge_ratios(X_test)

    # Save the model
    model.save_model('deep_hedging_model.h5')
    print("\nModel saved to 'deep_hedging_model.h5'")

if __name__ == "__main__":
    main()
