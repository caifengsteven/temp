import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import math
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class TimeSeriesTransformer:
    def __init__(self, 
                 sequence_length=32, 
                 d_model=16, 
                 num_heads=8, 
                 num_transformer_blocks=6,
                 ff_dim_factor=4, 
                 mlp_units=[10], 
                 mlp_dropout=0.25,
                 dropout=0.25, 
                 n_classes=7,
                 use_positional_encoding=False):
        """
        Initialize the Time Series Transformer model.
        
        Parameters:
        -----------
        sequence_length : int
            Length of input sequences
        d_model : int
            Dimension of the model (embedding dimension)
        num_heads : int
            Number of attention heads
        num_transformer_blocks : int
            Number of transformer blocks to stack
        ff_dim_factor : int
            Multiplier for the feed-forward network dimension
        mlp_units : list
            Units in the MLP layers after the transformer blocks
        mlp_dropout : float
            Dropout rate for the MLP layers
        dropout : float
            Dropout rate for the transformer blocks
        n_classes : int
            Number of classification buckets
        use_positional_encoding : bool
            Whether to use positional encoding
        """
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_transformer_blocks = num_transformer_blocks
        self.ff_dim_factor = ff_dim_factor
        self.ff_dim = ff_dim_factor * d_model
        self.mlp_units = mlp_units
        self.mlp_dropout = mlp_dropout
        self.dropout = dropout
        self.n_classes = n_classes
        self.use_positional_encoding = use_positional_encoding
        self.head_size = 64
        
        # Build the model
        self.model = self._build_model()
        
    def _build_model(self):
        """Build the transformer model for time series forecasting."""
        inputs = keras.Input(shape=(self.sequence_length, self.d_model))
        x = inputs
        
        # Add positional encoding if enabled
        if self.use_positional_encoding:
            x = x + self._get_positional_encoding()
        
        # Create multiple transformer blocks
        for _ in range(self.num_transformer_blocks):
            # Normalization before multi-head attention (as per the paper)
            x = layers.LayerNormalization(axis=-1, epsilon=1e-6)(x)
            
            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                key_dim=self.head_size, 
                num_heads=self.num_heads, 
                dropout=0
            )(x, x)
            attention_output = layers.Dropout(self.dropout)(attention_output)
            
            # Skip connection
            x = attention_output + x
            
            # Feed-forward network
            ffn_input = layers.LayerNormalization(axis=-1, epsilon=1e-6)(x)
            ffn = layers.Conv1D(filters=self.ff_dim, kernel_size=1, activation="relu")(ffn_input)
            ffn = layers.Dropout(self.dropout)(ffn)
            ffn = layers.Conv1D(filters=self.d_model, kernel_size=1)(ffn)
            
            # Skip connection
            x = ffn + x
        
        # Final normalization
        x = layers.LayerNormalization(axis=-1, epsilon=1e-6)(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        
        # MLP for classification
        for dim in self.mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(self.mlp_dropout)(x)
            
        # Output layer
        outputs = layers.Dense(self.n_classes, activation="softmax")(x)
        
        return keras.Model(inputs, outputs)
    
    def _get_positional_encoding(self):
        """Generate positional encoding as described in the paper."""
        positions = np.arange(self.sequence_length)[:, np.newaxis]
        dim_indices = np.arange(self.d_model)[np.newaxis, :]
        
        # Only use even dimensions for the sinusoidal encoding
        angle_rates = 1 / np.power(10000, (2 * (dim_indices // 2)) / self.d_model)
        angle_rads = positions * angle_rates
        
        # Apply sin to even indices
        encoding = np.zeros((self.sequence_length, self.d_model))
        encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # Apply cos to odd indices
        encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        return tf.cast(encoding, dtype=tf.float32)
    
    def compile_model(self, learning_rate=1e-3):
        """Compile the model with appropriate loss and metrics."""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy"]
        )
        
    def fit(self, x_train, y_train, epochs=30, batch_size=64, validation_split=0.2):
        """Train the model on the provided data."""
        return self.model.fit(
            x_train,
            y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ]
        )
    
    def evaluate(self, x_test, y_test):
        """Evaluate the model on test data."""
        return self.model.evaluate(x_test, y_test)
    
    def predict(self, x):
        """Make predictions using the trained model."""
        return self.model.predict(x)
    
    def summary(self):
        """Display model summary."""
        return self.model.summary()


def embed_time_series(y, d_model):
    """
    Embed 1D time series into d-dimensional space using polynomial expansion.
    φ(y) = (y, y²/2, y³/3!, ..., yᵈ/d!)
    """
    embedded = np.zeros((len(y), d_model))
    
    for i in range(len(y)):
        for j in range(d_model):
            embedded[i, j] = (y[i] ** (j+1)) / math.factorial(j+1)
            
    return embedded

def create_sequences(data, sequence_length):
    """Create sequences of fixed length from the data."""
    sequences = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i+sequence_length])
    return np.array(sequences)

def create_dataset(data, sequence_length, d_model):
    """
    Create a dataset of embedded sequences and target values.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Time series data
    sequence_length : int
        Length of each sequence
    d_model : int
        Embedding dimension
    
    Returns:
    --------
    X : numpy.ndarray
        Embedded sequences of shape (n_samples, sequence_length, d_model)
    y : numpy.ndarray
        Target values
    """
    # Embed the time series
    embedded_data = embed_time_series(data, d_model)
    
    # Create sequences
    X = create_sequences(embedded_data, sequence_length)
    
    # The target is the value following each sequence
    y = data[sequence_length:]
    
    return X, y

def create_buckets(y_train, n_classes=7):
    """
    Create bucket boundaries based on quantiles of the training data.
    
    Parameters:
    -----------
    y_train : numpy.ndarray
        Training target values
    n_classes : int
        Number of buckets/classes
    
    Returns:
    --------
    bucket_boundaries : numpy.ndarray
        Boundaries of the buckets
    """
    quantiles = np.linspace(0, 1, n_classes+1)[1:-1]
    bucket_boundaries = np.quantile(y_train, quantiles)
    return bucket_boundaries

def assign_buckets(y, bucket_boundaries):
    """
    Assign each value in y to a bucket based on the boundaries.
    
    Parameters:
    -----------
    y : numpy.ndarray
        Target values
    bucket_boundaries : numpy.ndarray
        Boundaries of the buckets
    
    Returns:
    --------
    bucket_indices : numpy.ndarray
        Bucket indices for each value in y
    """
    # Add -inf and inf to the boundaries
    extended_boundaries = np.concatenate([[-np.inf], bucket_boundaries, [np.inf]])
    
    # Assign each value to a bucket
    bucket_indices = np.zeros_like(y, dtype=int)
    for i in range(len(y)):
        for j in range(len(extended_boundaries) - 1):
            if extended_boundaries[j] < y[i] <= extended_boundaries[j+1]:
                bucket_indices[i] = j
                break
    
    return bucket_indices

def one_hot_encode(bucket_indices, n_classes):
    """One-hot encode the bucket indices."""
    return tf.keras.utils.to_categorical(bucket_indices, num_classes=n_classes)

def generate_ornstein_uhlenbeck(n_steps, theta=1.0, mu=0.0, sigma=1.0, dt=1.0, h0=0.0):
    """
    Generate a trajectory of the Ornstein-Uhlenbeck process.
    
    Parameters:
    -----------
    n_steps : int
        Number of steps to simulate
    theta : float
        Mean reversion speed
    mu : float
        Mean reversion level
    sigma : float
        Volatility
    dt : float
        Time step
    h0 : float
        Initial value
    
    Returns:
    --------
    h : numpy.ndarray
        Hidden state values
    y : numpy.ndarray
        Observable values (differences of h)
    """
    # Initialize arrays
    h = np.zeros(n_steps + 1)
    h[0] = h0
    
    # Generate the process
    for i in range(n_steps):
        epsilon = np.random.normal(0, 1)
        h[i+1] = h[i] + theta * (mu - h[i]) * dt + sigma * np.sqrt(dt) * epsilon
    
    # Calculate the observable values (differences)
    y = np.diff(h)
    
    return h, y

def calculate_target_probabilities(h, bucket_boundaries, theta=1.0, mu=0.0, sigma=1.0, dt=1.0):
    """
    Calculate the target probabilities P(y_{i+1} ∈ Bj | h_i).
    
    Parameters:
    -----------
    h : numpy.ndarray
        Hidden state values
    bucket_boundaries : numpy.ndarray
        Boundaries of the buckets
    theta : float
        Mean reversion speed
    mu : float
        Mean reversion level
    sigma : float
        Volatility
    dt : float
        Time step
    
    Returns:
    --------
    target_probs : numpy.ndarray
        Target probabilities for each hidden state and bucket
    """
    n_samples = len(h)
    n_classes = len(bucket_boundaries) + 1
    
    # Initialize the target probabilities
    target_probs = np.zeros((n_samples, n_classes))
    
    # Add -inf and inf to the boundaries
    extended_boundaries = np.concatenate([[-np.inf], bucket_boundaries, [np.inf]])
    
    # Calculate the probabilities for each hidden state and bucket
    for i in range(n_samples):
        h_i = h[i]
        
        # The mean of the next hidden state
        mean_next = h_i + theta * (mu - h_i) * dt
        
        # The standard deviation of the next hidden state
        std_next = sigma * np.sqrt(dt)
        
        for j in range(n_classes):
            # Calculate the probability of the next observable value being in bucket j
            lower_bound = extended_boundaries[j] + h_i
            upper_bound = extended_boundaries[j+1] + h_i
            
            # Probability of being in the bucket
            prob_lower = norm.cdf(lower_bound, mean_next, std_next)
            prob_upper = norm.cdf(upper_bound, mean_next, std_next)
            target_probs[i, j] = prob_upper - prob_lower
    
    return target_probs

def plot_results(history, y_true, y_pred, bucket_boundaries, title="Model Performance"):
    """Plot the training history and prediction results."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training and validation loss
    axs[0, 0].plot(history.history['loss'], label='Train Loss')
    axs[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axs[0, 0].set_title('Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    
    # Plot training and validation accuracy
    axs[0, 1].plot(history.history['categorical_accuracy'], label='Train Accuracy')
    axs[0, 1].plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
    axs[0, 1].set_title('Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()
    
    # Plot predicted probabilities for each bucket
    for i in range(min(5, len(bucket_boundaries) + 1)):
        axs[1, 0].scatter(y_true, y_pred[:, i], alpha=0.5, label=f'Bucket {i+1}')
    axs[1, 0].set_title('Predicted Probabilities vs True Values')
    axs[1, 0].set_xlabel('True Value')
    axs[1, 0].set_ylabel('Predicted Probability')
    axs[1, 0].legend()
    
    # Plot the distribution of predictions
    true_buckets = assign_buckets(y_true, bucket_boundaries)
    pred_buckets = np.argmax(y_pred, axis=1)
    
    axs[1, 1].hist([true_buckets, pred_buckets], bins=len(bucket_boundaries)+1, 
                 label=['True', 'Predicted'], alpha=0.7)
    axs[1, 1].set_title('Distribution of Buckets')
    axs[1, 1].set_xlabel('Bucket')
    axs[1, 1].set_ylabel('Count')
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=16)
    plt.show()

def evaluate_model(y_true, y_pred, bucket_boundaries):
    """Evaluate the model performance."""
    true_buckets = assign_buckets(y_true, bucket_boundaries)
    pred_buckets = np.argmax(y_pred, axis=1)
    
    accuracy = accuracy_score(true_buckets, pred_buckets)
    report = classification_report(true_buckets, pred_buckets)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    return accuracy, report

# Import necessary library for normal distribution CDF
from scipy.stats import norm

def experiment_ornstein_uhlenbeck():
    """Run the experiment on synthetic Ornstein-Uhlenbeck data."""
    print("Running Ornstein-Uhlenbeck experiment...")
    
    # Parameters
    n_steps = 24131  # Same as the S&P500 data in the paper
    sequence_length = 32
    d_model = 16
    n_classes = 7
    
    # Generate the synthetic data
    h, y = generate_ornstein_uhlenbeck(n_steps, theta=1.0, mu=0.0, sigma=1.0, dt=1.0, h0=0.0)
    
    # Create the dataset
    X, y_target = create_dataset(y, sequence_length, d_model)
    
    # Create bucket boundaries based on the training set
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y_target[:train_size], y_target[train_size:]
    
    bucket_boundaries = create_buckets(y_train, n_classes)
    
    # Assign buckets and one-hot encode
    y_train_buckets = assign_buckets(y_train, bucket_boundaries)
    y_test_buckets = assign_buckets(y_test, bucket_boundaries)
    
    y_train_onehot = one_hot_encode(y_train_buckets, n_classes)
    y_test_onehot = one_hot_encode(y_test_buckets, n_classes)
    
    # Create and train the model
    model = TimeSeriesTransformer(
        sequence_length=sequence_length,
        d_model=d_model,
        n_classes=n_classes,
        use_positional_encoding=False  # As per the paper
    )
    
    model.compile_model()
    model.summary()
    
    history = model.fit(X_train, y_train_onehot, epochs=30, batch_size=64)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test_onehot)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    # Calculate target probabilities
    h_test = h[sequence_length+train_size:train_size+len(y_test)+sequence_length]
    target_probs = calculate_target_probabilities(h_test, bucket_boundaries)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate cross-entropy between target and predicted probabilities
    ce_pred = -np.sum(target_probs * np.log(y_pred + 1e-10)) / len(y_test)
    ce_target = -np.sum(target_probs * np.log(target_probs + 1e-10)) / len(y_test)
    
    print(f"Cross-entropy between target and predicted: {ce_pred:.4f}")
    print(f"Cross-entropy between target and target (optimal): {ce_target:.4f}")
    
    # Plot the results
    plot_results(history, y_test, y_pred, bucket_boundaries, 
                title="Ornstein-Uhlenbeck Process Prediction")
    
    # Plot the target probabilities vs. predicted probabilities for each bucket
    plt.figure(figsize=(15, 10))
    for i in range(n_classes):
        plt.subplot(3, 3, i+1)
        plt.scatter(h_test, target_probs[:, i], alpha=0.5, label='Target', color='blue')
        plt.scatter(h_test, y_pred[:, i], alpha=0.5, label='Predicted', color='red')
        plt.title(f'Bucket {i+1}')
        plt.xlabel('Hidden State')
        plt.ylabel('Probability')
        plt.legend()
    
    plt.tight_layout()
    plt.suptitle("Target vs. Predicted Probabilities", y=1.02, fontsize=16)
    plt.show()
    
    return model, history, y_test, y_pred, bucket_boundaries

def experiment_sp500_returns():
    """Run the experiment on S&P500 daily returns."""
    print("Running S&P500 returns experiment...")
    
    # Parameters
    sequence_length = 32
    d_model = 16
    n_classes = 7
    
    # Download S&P500 data
    sp500 = yf.download('^GSPC', start='1927-12-30', end='2024-02-01')
    print(f"Downloaded {len(sp500)} days of S&P500 data")
    
    # Calculate log returns
    sp500['LogReturn'] = np.log(sp500['Close']) - np.log(sp500['Close'].shift(1))
    sp500 = sp500.dropna()
    
    returns = sp500['LogReturn'].values
    
    # Create the dataset
    X, y_target = create_dataset(returns, sequence_length, d_model)
    
    # Create bucket boundaries based on the training set
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y_target[:train_size], y_target[train_size:]
    
    bucket_boundaries = create_buckets(y_train, n_classes)
    
    # Assign buckets and one-hot encode
    y_train_buckets = assign_buckets(y_train, bucket_boundaries)
    y_test_buckets = assign_buckets(y_test, bucket_boundaries)
    
    y_train_onehot = one_hot_encode(y_train_buckets, n_classes)
    y_test_onehot = one_hot_encode(y_test_buckets, n_classes)
    
    # Create and train the model
    model = TimeSeriesTransformer(
        sequence_length=sequence_length,
        d_model=d_model,
        n_classes=n_classes,
        use_positional_encoding=False
    )
    
    model.compile_model()
    
    history = model.fit(X_train, y_train_onehot, epochs=30, batch_size=64)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test_onehot)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Plot the results
    plot_results(history, y_test, y_pred, bucket_boundaries, 
                title="S&P500 Returns Prediction")
    
    # Evaluate the model
    evaluate_model(y_test, y_pred, bucket_boundaries)
    
    # Compare to naive classifier
    naive_y_pred = np.zeros((len(y_test), n_classes))
    for i in range(len(y_test)):
        last_value = np.mean(returns[train_size+i:train_size+i+sequence_length])
        last_bucket = assign_buckets(np.array([last_value]), bucket_boundaries)[0]
        naive_y_pred[i, last_bucket] = 1
    
    naive_accuracy = accuracy_score(y_test_buckets, np.argmax(naive_y_pred, axis=1))
    print(f"Naive classifier accuracy: {naive_accuracy:.4f}")
    
    return model, history, y_test, y_pred, bucket_boundaries

def experiment_sp500_quadratic_variation():
    """Run the experiment on S&P500 quadratic variation (squared returns)."""
    print("Running S&P500 quadratic variation experiment...")
    
    # Parameters
    sequence_length = 32
    d_model = 16
    n_classes = 7
    
    # Download S&P500 data
    sp500 = yf.download('^GSPC', start='1927-12-30', end='2024-02-01')
    print(f"Downloaded {len(sp500)} days of S&P500 data")
    
    # Calculate log returns and squared returns
    sp500['LogReturn'] = np.log(sp500['Close']) - np.log(sp500['Close'].shift(1))
    sp500['SquaredReturn'] = sp500['LogReturn'] ** 2
    sp500 = sp500.dropna()
    
    # Use log returns as input features but predict squared returns
    returns = sp500['LogReturn'].values
    squared_returns = sp500['SquaredReturn'].values
    
    # Create the dataset for log returns (input)
    X_returns, _ = create_dataset(returns, sequence_length, d_model)
    
    # Target is the squared return after each sequence
    y_target = squared_returns[sequence_length:]
    
    # Create bucket boundaries based on the training set
    train_size = int(0.8 * len(X_returns))
    X_train, X_test = X_returns[:train_size], X_returns[train_size:]
    y_train, y_test = y_target[:train_size], y_target[train_size:]
    
    bucket_boundaries = create_buckets(y_train, n_classes)
    
    # Assign buckets and one-hot encode
    y_train_buckets = assign_buckets(y_train, bucket_boundaries)
    y_test_buckets = assign_buckets(y_test, bucket_boundaries)
    
    y_train_onehot = one_hot_encode(y_train_buckets, n_classes)
    y_test_onehot = one_hot_encode(y_test_buckets, n_classes)
    
    # Create and train the model
    model = TimeSeriesTransformer(
        sequence_length=sequence_length,
        d_model=d_model,
        n_classes=n_classes,
        use_positional_encoding=False
    )
    
    model.compile_model()
    
    history = model.fit(X_train, y_train_onehot, epochs=50, batch_size=64)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test_onehot)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Plot the results
    plot_results(history, y_test, y_pred, bucket_boundaries, 
                title="S&P500 Quadratic Variation Prediction")
    
    # Evaluate the model
    evaluate_model(y_test, y_pred, bucket_boundaries)
    
    # Compare to naive classifier (using mean of previous squared returns)
    naive_y_pred = np.zeros((len(y_test), n_classes))
    for i in range(len(y_test)):
        last_values = squared_returns[train_size+i:train_size+i+sequence_length]
        last_value = np.mean(last_values)
        last_bucket = assign_buckets(np.array([last_value]), bucket_boundaries)[0]
        naive_y_pred[i, last_bucket] = 1
    
    naive_accuracy = accuracy_score(y_test_buckets, np.argmax(naive_y_pred, axis=1))
    print(f"Naive classifier accuracy: {naive_accuracy:.4f}")
    
    return model, history, y_test, y_pred, bucket_boundaries

# Run the experiments
if __name__ == "__main__":
    # Ornstein-Uhlenbeck experiment
    ou_model, ou_history, ou_y_test, ou_y_pred, ou_bucket_boundaries = experiment_ornstein_uhlenbeck()
    
    # S&P500 returns experiment
    sp_model, sp_history, sp_y_test, sp_y_pred, sp_bucket_boundaries = experiment_sp500_returns()
    
    # S&P500 quadratic variation experiment
    qv_model, qv_history, qv_y_test, qv_y_pred, qv_bucket_boundaries = experiment_sp500_quadratic_variation()
    
    # Save the trained models
    current_date = datetime.now().strftime("%Y%m%d")
    ou_model.model.save(f"ou_transformer_{current_date}.h5")
    sp_model.model.save(f"sp500_returns_transformer_{current_date}.h5")
    qv_model.model.save(f"sp500_quadratic_variation_transformer_{current_date}.h5")
    
    print("All experiments completed and models saved.")

    