import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class RVRAE:
    """
    Recurrent Variational Autoencoder for Dynamic Factor Model
    
    This model combines a variational autoencoder with recurrent layers
    to capture both the temporal dependencies in stock data and handle
    noisy market conditions.
    """
    
    def __init__(self, input_dim, char_dim, latent_dim=5, timesteps=12):
        """
        Initialize the RVRAE model
        
        Parameters:
        -----------
        input_dim : int
            Dimension of input data (number of stocks)
        char_dim : int
            Dimension of firm characteristics
        latent_dim : int
            Dimension of latent space (number of factors)
        timesteps : int
            Number of time steps for the RNN
        """
        self.input_dim = input_dim
        self.char_dim = char_dim
        self.latent_dim = latent_dim
        self.timesteps = timesteps
        self.encoder = None
        self.decoder = None
        self.beta_network = None
        self.model = None
        
    def sampling(self, args):
        """
        Reparameterization trick: Sample from the latent space
        
        Parameters:
        -----------
        args : tuple
            Mean and log variance of the latent distribution
            
        Returns:
        --------
        z : tensor
            Sampled latent vector
        """
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def build_encoder(self):
        """
        Build the encoder network
        
        Returns:
        --------
        encoder : Model
            Encoder model
        """
        # Input layer for stock returns
        inputs = layers.Input(shape=(self.timesteps, self.input_dim), name='encoder_input')
        
        # RNN layers
        x = layers.LSTM(128, return_sequences=True)(inputs)
        x = layers.LSTM(64, return_sequences=False)(x)
        
        # Latent space parameters
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        
        # Reparameterization trick
        z = layers.Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])
        
        # Define encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        
        self.encoder = encoder
        return encoder
    
    def build_decoder(self):
        """
        Build the decoder network
        
        Returns:
        --------
        decoder : Model
            Decoder model
        """
        # Input layer for latent variables
        latent_inputs = layers.Input(shape=(self.latent_dim,), name='z_sampling')
        
        # Dense layers
        x = layers.Dense(64, activation='relu')(latent_inputs)
        x = layers.Dense(64, activation='relu')(x)
        
        # Output layer
        factors = layers.Dense(self.latent_dim, activation='tanh', name='factors')(x)
        
        # Define decoder model
        decoder = Model(latent_inputs, factors, name='decoder')
        
        self.decoder = decoder
        return decoder
    
    def build_beta_network(self):
        """
        Build the beta network for factor exposures
        
        Returns:
        --------
        beta_network : Model
            Beta network model
        """
        # Input layer for firm characteristics
        inputs = layers.Input(shape=(self.timesteps, self.char_dim), name='beta_input')
        
        # LSTM layers for temporal dependencies
        x = layers.LSTM(128, return_sequences=True)(inputs)
        x = layers.LSTM(64, return_sequences=False)(x)
        
        # Output layer for factor exposures
        beta_output = layers.Dense(self.latent_dim, name='beta_output')(x)
        
        # Define beta network model
        beta_network = Model(inputs, beta_output, name='beta_network')
        
        self.beta_network = beta_network
        return beta_network
    
    def build_model(self):
        """
        Build the complete RVRAE model
        
        Returns:
        --------
        model : Model
            Complete RVRAE model
        """
        # Build the encoder, decoder, and beta network
        encoder = self.build_encoder()
        decoder = self.build_decoder()
        beta_network = self.build_beta_network()
        
        # Input layers
        returns_input = layers.Input(shape=(self.timesteps, self.input_dim), name='returns_input')
        chars_input = layers.Input(shape=(self.timesteps, self.char_dim), name='chars_input')
        
        # Get latent representations
        z_mean, z_log_var, z = encoder(returns_input)
        
        # Get factor exposures from beta network
        beta = beta_network(chars_input)
        
        # Get factors from decoder
        factors = decoder(z)
        
        # Predict returns by multiplying beta and factors
        predicted_returns = layers.Dot(axes=1, name='returns_prediction')([beta, factors])
        
        # Define model with two inputs and one output
        model = Model([returns_input, chars_input], predicted_returns, name='rvrae')
        
        # Add KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        model.add_loss(kl_loss)
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate for the optimizer
        """
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse'
        )
    
    def fit(self, X_returns, X_chars, y, validation_data=None, epochs=100, batch_size=32, verbose=1):
        """
        Fit the model to the data
        
        Parameters:
        -----------
        X_returns : array-like
            Input stock returns with shape (samples, timesteps, stocks)
        X_chars : array-like
            Input firm characteristics with shape (samples, timesteps, characteristics)
        y : array-like
            Target stock returns with shape (samples, stocks)
        validation_data : tuple
            Validation data in the form ((X_returns_val, X_chars_val), y_val)
        epochs : int
            Number of epochs to train
        batch_size : int
            Batch size for training
        verbose : int
            Verbosity mode
            
        Returns:
        --------
        history : History
            Training history
        """
        # Set up early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Fit the model
        history = self.model.fit(
            [X_returns, X_chars],
            y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=verbose
        )
        
        return history
    
    def predict(self, X_returns, X_chars):
        """
        Predict using the model
        
        Parameters:
        -----------
        X_returns : array-like
            Input stock returns with shape (samples, timesteps, stocks)
        X_chars : array-like
            Input firm characteristics with shape (samples, timesteps, characteristics)
            
        Returns:
        --------
        predictions : array
            Predicted stock returns
        """
        return self.model.predict([X_returns, X_chars])
    
    def evaluate(self, X_returns, X_chars, y):
        """
        Evaluate the model
        
        Parameters:
        -----------
        X_returns : array-like
            Input stock returns with shape (samples, timesteps, stocks)
        X_chars : array-like
            Input firm characteristics with shape (samples, timesteps, characteristics)
        y : array-like
            Target stock returns with shape (samples, stocks)
            
        Returns:
        --------
        metrics : dict
            Evaluation metrics
        """
        # Predict
        y_pred = self.predict(X_returns, X_chars)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Calculate total R^2 as defined in the paper
        total_r2 = 1 - np.sum((y - y_pred)**2) / np.sum(y**2)
        
        # Calculate predictive R^2 (simplified version)
        y_mean = np.mean(y, axis=0)
        predictive_r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y_mean)**2)
        
        return {
            'mse': mse,
            'r2': r2,
            'total_r2': total_r2,
            'predictive_r2': predictive_r2
        }
    
    def get_latent_factors(self, X_returns):
        """
        Get latent factors from the encoder
        
        Parameters:
        -----------
        X_returns : array-like
            Input stock returns with shape (samples, timesteps, stocks)
            
        Returns:
        --------
        factors : array
            Latent factors
        """
        _, _, z = self.encoder.predict(X_returns)
        factors = self.decoder.predict(z)
        return factors


class BaselineModels:
    """
    Baseline models for comparison
    """
    
    @staticmethod
    def ipca(X_returns, X_chars, y, latent_dim=5):
        """
        Instrumented Principal Component Analysis (IPCA)
        
        A simplified version of the IPCA model described in the paper.
        
        Parameters:
        -----------
        X_returns : array-like
            Input stock returns with shape (samples, timesteps, stocks)
        X_chars : array-like
            Input firm characteristics with shape (samples, timesteps, characteristics)
        y : array-like
            Target stock returns with shape (samples, stocks)
        latent_dim : int
            Number of factors
            
        Returns:
        --------
        model : dict
            IPCA model
        """
        # Reshape data
        n_samples = X_returns.shape[0]
        n_stocks = X_returns.shape[2]
        n_chars = X_chars.shape[2]
        
        # We'll use the last time step for firm characteristics
        X_chars_last = X_chars[:, -1, :]
        
        # Initialize factors and betas
        factors = np.random.randn(n_samples, latent_dim)
        betas = np.zeros((n_stocks, latent_dim))
        
        # Iterate to find factors and betas
        for _ in range(10):  # Simplified iteration
            # Update betas
            for s in range(n_stocks):
                X_s = X_chars_last[:, s % n_chars].reshape(-1, 1)  # Use char % n_chars as a proxy
                betas[s] = np.linalg.lstsq(X_s * factors, y[:, s], rcond=None)[0].flatten()
            
            # Update factors
            for t in range(n_samples):
                factors[t] = np.linalg.lstsq(betas, y[t], rcond=None)[0]
        
        # Predict
        y_pred = np.zeros_like(y)
        for t in range(n_samples):
            y_pred[t] = betas @ factors[t]
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        total_r2 = 1 - np.sum((y - y_pred)**2) / np.sum(y**2)
        y_mean = np.mean(y, axis=0)
        predictive_r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y_mean)**2)
        
        return {
            'model_name': 'IPCA',
            'factors': factors,
            'betas': betas,
            'y_pred': y_pred,
            'mse': mse,
            'r2': r2,
            'total_r2': total_r2,
            'predictive_r2': predictive_r2
        }
    
    @staticmethod
    def alstm(X_returns, X_chars, y, latent_dim=5, epochs=50, batch_size=32):
        """
        LSTM with Attention for stock returns prediction
        
        Parameters:
        -----------
        X_returns : array-like
            Input stock returns with shape (samples, timesteps, stocks)
        X_chars : array-like
            Input firm characteristics with shape (samples, timesteps, characteristics)
        y : array-like
            Target stock returns with shape (samples, stocks)
        latent_dim : int
            Number of factors
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
            
        Returns:
        --------
        model : dict
            ALSTM model
        """
        # Define input shape
        input_shape_returns = X_returns.shape[1:]
        input_shape_chars = X_chars.shape[1:]
        
        # Define the model
        returns_input = layers.Input(shape=input_shape_returns)
        chars_input = layers.Input(shape=input_shape_chars)
        
        # Returns branch
        x1 = layers.LSTM(128, return_sequences=True)(returns_input)
        
        # Attention mechanism
        attention = layers.Dense(1, activation='tanh')(x1)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(128)(attention)
        attention = layers.Permute([2, 1])(attention)
        
        # Apply attention
        x1 = layers.Multiply()([x1, attention])
        x1 = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(x1)
        
        # Chars branch
        x2 = layers.LSTM(128, return_sequences=False)(chars_input)
        
        # Combine branches
        x = layers.Concatenate()([x1, x2])
        x = layers.Dense(64, activation='relu')(x)
        
        # Output
        output = layers.Dense(y.shape[1])(x)
        
        # Create model
        model = Model([returns_input, chars_input], output)
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Train model
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # Split data for validation
        val_split = 0.2
        n_val = int(X_returns.shape[0] * val_split)
        
        X_returns_train, X_returns_val = X_returns[:-n_val], X_returns[-n_val:]
        X_chars_train, X_chars_val = X_chars[:-n_val], X_chars[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]
        
        model.fit(
            [X_returns_train, X_chars_train],
            y_train,
            validation_data=([X_returns_val, X_chars_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Predict
        y_pred = model.predict([X_returns, X_chars])
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        total_r2 = 1 - np.sum((y - y_pred)**2) / np.sum(y**2)
        y_mean = np.mean(y, axis=0)
        predictive_r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y_mean)**2)
        
        return {
            'model_name': 'ALSTM',
            'model': model,
            'y_pred': y_pred,
            'mse': mse,
            'r2': r2,
            'total_r2': total_r2,
            'predictive_r2': predictive_r2
        }
    
    @staticmethod
    def factorVAE(X_returns, X_chars, y, latent_dim=5, epochs=50, batch_size=32):
        """
        Probabilistic Dynamic Factor Model Based on Variational Autoencoder
        
        Parameters:
        -----------
        X_returns : array-like
            Input stock returns with shape (samples, timesteps, stocks)
        X_chars : array-like
            Input firm characteristics with shape (samples, timesteps, characteristics)
        y : array-like
            Target stock returns with shape (samples, stocks)
        latent_dim : int
            Number of factors
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
            
        Returns:
        --------
        model : dict
            FactorVAE model
        """
        # Define input shapes
        input_shape_returns = X_returns.shape[1:]
        input_shape_chars = X_chars.shape[1:]
        
        # Encoder
        returns_input = layers.Input(shape=input_shape_returns)
        
        # Flatten the time dimension
        x = layers.Reshape((input_shape_returns[0] * input_shape_returns[1],))(returns_input)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        
        # Latent space parameters
        z_mean = layers.Dense(latent_dim)(x)
        z_log_var = layers.Dense(latent_dim)(x)
        
        # Sampling function
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        # Reparameterization trick
        z = layers.Lambda(sampling)([z_mean, z_log_var])
        
        # Beta network
        chars_input = layers.Input(shape=input_shape_chars)
        
        # Reshape and compress
        c = layers.Reshape((input_shape_chars[0] * input_shape_chars[1],))(chars_input)
        c = layers.Dense(128, activation='relu')(c)
        c = layers.Dense(64, activation='relu')(c)
        
        # Beta output
        beta = layers.Dense(latent_dim)(c)
        
        # Decoder
        decoder_input = layers.Input(shape=(latent_dim,))
        d = layers.Dense(64, activation='relu')(decoder_input)
        d = layers.Dense(latent_dim, activation='tanh')(d)
        
        # Define models
        encoder = Model(returns_input, [z_mean, z_log_var, z], name='encoder')
        decoder = Model(decoder_input, d, name='decoder')
        beta_network = Model(chars_input, beta, name='beta_network')
        
        # Encode returns
        _, _, z = encoder(returns_input)
        
        # Get factors
        factors = decoder(z)
        
        # Get betas
        beta = beta_network(chars_input)
        
        # Predict returns
        predicted_returns = layers.Dot(axes=1)([beta, factors])
        
        # Define final model
        model = Model([returns_input, chars_input], predicted_returns)
        
        # Add KL loss
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        model.add_loss(kl_loss)
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Train model
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # Split data for validation
        val_split = 0.2
        n_val = int(X_returns.shape[0] * val_split)
        
        X_returns_train, X_returns_val = X_returns[:-n_val], X_returns[-n_val:]
        X_chars_train, X_chars_val = X_chars[:-n_val], X_chars[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]
        
        model.fit(
            [X_returns_train, X_chars_train],
            y_train,
            validation_data=([X_returns_val, X_chars_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Predict
        y_pred = model.predict([X_returns, X_chars])
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        total_r2 = 1 - np.sum((y - y_pred)**2) / np.sum(y**2)
        y_mean = np.mean(y, axis=0)
        predictive_r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y_mean)**2)
        
        return {
            'model_name': 'FactorVAE',
            'encoder': encoder,
            'decoder': decoder,
            'beta_network': beta_network,
            'model': model,
            'y_pred': y_pred,
            'mse': mse,
            'r2': r2,
            'total_r2': total_r2,
            'predictive_r2': predictive_r2
        }


def generate_synthetic_data(n_samples=1000, n_stocks=50, n_chars=46, n_timesteps=12, n_factors=5, seed=42):
    """
    Generate synthetic stock market data
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_stocks : int
        Number of stocks
    n_chars : int
        Number of firm characteristics
    n_timesteps : int
        Number of time steps
    n_factors : int
        Number of latent factors
    seed : int
        Random seed
        
    Returns:
    --------
    data : dict
        Dictionary containing the generated data
    """
    np.random.seed(seed)
    
    # Generate latent factors
    factors = np.random.randn(n_samples, n_factors)
    
    # Generate firm characteristics
    chars = np.random.randn(n_samples, n_timesteps, n_chars)
    
    # Generate factor exposures (betas) as a function of firm characteristics
    # Here we use a simple linear relationship
    betas = np.zeros((n_samples, n_stocks, n_factors))
    for t in range(n_samples):
        for s in range(n_stocks):
            # Use some characteristics to generate betas
            char_subset = chars[t, -1, :min(n_chars, 10)]
            betas[t, s] = 0.1 * char_subset[:n_factors] + np.random.randn(n_factors) * 0.05
    
    # Generate stock returns
    returns = np.zeros((n_samples, n_stocks))
    for t in range(n_samples):
        returns[t] = betas[t] @ factors[t] + np.random.randn(n_stocks) * 0.02
    
    # Generate historical returns for input to RNN
    historical_returns = np.zeros((n_samples, n_timesteps, n_stocks))
    for t in range(n_samples):
        if t < n_timesteps:
            # For initial samples, generate random history
            historical_returns[t] = np.random.randn(n_timesteps, n_stocks) * 0.05
        else:
            # For later samples, use actual previous returns
            for i in range(n_timesteps):
                historical_returns[t, i] = returns[t - n_timesteps + i]
    
    return {
        'factors': factors,
        'chars': chars,
        'betas': betas,
        'returns': returns,
        'historical_returns': historical_returns
    }


def calculate_sharpe_ratio(returns, risk_free_rate=0.001):
    """
    Calculate the Sharpe ratio
    
    Parameters:
    -----------
    returns : array-like
        Returns
    risk_free_rate : float
        Risk-free rate (monthly)
        
    Returns:
    --------
    sharpe : float
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns)


def portfolio_simulation(y_true, y_pred, n_top=10, transaction_cost=0.0030):
    """
    Simulate a portfolio based on predicted returns
    
    Parameters:
    -----------
    y_true : array-like
        True returns
    y_pred : array-like
        Predicted returns
    n_top : int
        Number of stocks to include in the portfolio
    transaction_cost : float
        Transaction cost as a fraction of the position size
        
    Returns:
    --------
    results : dict
        Dictionary containing the simulation results
    """
    n_periods = y_true.shape[0]
    
    # Initialize portfolio
    portfolio_returns = np.zeros(n_periods)
    portfolio_weights = np.zeros_like(y_true)
    turnover = np.zeros(n_periods)
    
    for t in range(n_periods):
        # Rank stocks by predicted returns
        ranks = np.argsort(-y_pred[t])
        
        # Select top n_top stocks
        selected = ranks[:n_top]
        
        # Equal weighting
        new_weights = np.zeros(y_true.shape[1])
        new_weights[selected] = 1.0 / n_top
        
        # Calculate turnover
        if t > 0:
            turnover[t] = np.sum(np.abs(new_weights - portfolio_weights[t-1]))
            
            # Apply transaction costs
            transaction_costs = turnover[t] * transaction_cost
        else:
            transaction_costs = np.sum(new_weights) * transaction_cost
        
        # Update weights
        portfolio_weights[t] = new_weights
        
        # Calculate portfolio return (with transaction costs)
        portfolio_returns[t] = np.sum(portfolio_weights[t] * y_true[t]) - transaction_costs
    
    # Calculate metrics
    cumulative_return = np.prod(1 + portfolio_returns) - 1
    annualized_return = (1 + cumulative_return) ** (12 / n_periods) - 1
    annualized_volatility = np.std(portfolio_returns) * np.sqrt(12)
    sharpe_ratio = calculate_sharpe_ratio(portfolio_returns)
    sharpe_ratio_annualized = sharpe_ratio * np.sqrt(12)
    max_drawdown = np.max(np.maximum.accumulate(np.cumprod(1 + portfolio_returns)) - np.cumprod(1 + portfolio_returns)) / np.max(np.maximum.accumulate(np.cumprod(1 + portfolio_returns)))
    
    return {
        'portfolio_returns': portfolio_returns,
        'cumulative_return': cumulative_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio,
        'sharpe_ratio_annualized': sharpe_ratio_annualized,
        'max_drawdown': max_drawdown,
        'turnover': np.mean(turnover[1:])
    }


def rank_ic(y_true, y_pred):
    """
    Calculate the Rank Information Coefficient
    
    Parameters:
    -----------
    y_true : array-like
        True returns
    y_pred : array-like
        Predicted returns
        
    Returns:
    --------
    rank_ic : float
        Rank Information Coefficient
    """
    n_periods = y_true.shape[0]
    rank_ic_values = np.zeros(n_periods)
    
    for t in range(n_periods):
        # Calculate ranks
        true_ranks = np.argsort(np.argsort(y_true[t]))
        pred_ranks = np.argsort(np.argsort(y_pred[t]))
        
        # Calculate correlation
        rank_ic_values[t] = np.corrcoef(true_ranks, pred_ranks)[0, 1]
    
    return np.mean(rank_ic_values)


def evaluate_model_performance(models_results, show_plots=True):
    """
    Evaluate and compare model performance
    
    Parameters:
    -----------
    models_results : dict
        Dictionary containing the results for each model
    show_plots : bool
        Whether to show plots
        
    Returns:
    --------
    summary : DataFrame
        Summary of model performance
    """
    # Extract model names
    model_names = list(models_results.keys())
    
    # Prepare summary dataframe
    summary = pd.DataFrame(
        index=model_names,
        columns=['Total R2', 'Predictive R2', 'MSE', 'Sharpe Ratio', 'Sharpe with Costs', 'Rank IC', 'Rank ICIR']
    )
    
    # Fill summary dataframe
    for model_name in model_names:
        results = models_results[model_name]
        summary.loc[model_name, 'Total R2'] = results['metrics']['total_r2'] * 100
        summary.loc[model_name, 'Predictive R2'] = results['metrics']['predictive_r2'] * 100
        summary.loc[model_name, 'MSE'] = results['metrics']['mse']
        summary.loc[model_name, 'Sharpe Ratio'] = results['portfolio']['sharpe_ratio_annualized']
        summary.loc[model_name, 'Sharpe with Costs'] = results['portfolio_with_costs']['sharpe_ratio_annualized']
        summary.loc[model_name, 'Rank IC'] = results['rank_ic']
        summary.loc[model_name, 'Rank ICIR'] = results['rank_ic'] / np.std(results['y_pred'].flatten())
    
    if show_plots:
        # Plot Total R2 and Predictive R2
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.barplot(x=summary.index, y='Total R2', data=summary)
        plt.title('Total R²')
        plt.ylabel('R² (%)')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        sns.barplot(x=summary.index, y='Predictive R2', data=summary)
        plt.title('Predictive R²')
        plt.ylabel('R² (%)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Plot Sharpe Ratios
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.barplot(x=summary.index, y='Sharpe Ratio', data=summary)
        plt.title('Sharpe Ratio (Without Transaction Costs)')
        plt.ylabel('Annualized Sharpe Ratio')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        sns.barplot(x=summary.index, y='Sharpe with Costs', data=summary)
        plt.title('Sharpe Ratio (With 30bps Transaction Costs)')
        plt.ylabel('Annualized Sharpe Ratio')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Plot cumulative returns
        plt.figure(figsize=(14, 7))
        
        for model_name in model_names:
            results = models_results[model_name]
            returns = results['portfolio_with_costs']['portfolio_returns']
            plt.plot(np.cumprod(1 + returns) - 1, label=model_name)
        
        plt.title('Cumulative Returns (With Transaction Costs)')
        plt.xlabel('Time Period')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return summary


def main():
    """
    Main function to run the experiment
    """
    print("Generating synthetic data...")
    
    # Generate synthetic data
    data = generate_synthetic_data(n_samples=500, n_stocks=50, n_chars=46, n_timesteps=12, n_factors=5)
    
    # Split data into train, validation, and test sets
    train_size = int(0.6 * data['returns'].shape[0])
    val_size = int(0.2 * data['returns'].shape[0])
    
    # Prepare data
    X_returns_train = data['historical_returns'][:train_size]
    X_chars_train = data['chars'][:train_size]
    y_train = data['returns'][:train_size]
    
    X_returns_val = data['historical_returns'][train_size:train_size + val_size]
    X_chars_val = data['chars'][train_size:train_size + val_size]
    y_val = data['returns'][train_size:train_size + val_size]
    
    X_returns_test = data['historical_returns'][train_size + val_size:]
    X_chars_test = data['chars'][train_size + val_size:]
    y_test = data['returns'][train_size + val_size:]
    
    print(f"Data shapes - Train: {X_returns_train.shape}, Val: {X_returns_val.shape}, Test: {X_returns_test.shape}")
    
    # Initialize models
    print("Training RVRAE model...")
    rvrae = RVRAE(
        input_dim=X_returns_train.shape[2],
        char_dim=X_chars_train.shape[2],
        latent_dim=5,
        timesteps=X_returns_train.shape[1]
    )
    rvrae.build_model()
    rvrae.compile_model()
    
    # Train RVRAE model
    history = rvrae.fit(
        X_returns_train, X_chars_train, y_train,
        validation_data=([X_returns_val, X_chars_val], y_val),
        epochs=50,
        batch_size=32,
        verbose=0
    )
    
    # Evaluate RVRAE model
    rvrae_metrics = rvrae.evaluate(X_returns_test, X_chars_test, y_test)
    rvrae_pred = rvrae.predict(X_returns_test, X_chars_test)
    
    # Train baseline models
    print("Training baseline models...")
    
    print("Training IPCA model...")
    ipca_results = BaselineModels.ipca(X_returns_test, X_chars_test, y_test)
    
    print("Training ALSTM model...")
    alstm_results = BaselineModels.alstm(X_returns_train, X_chars_train, y_train)
    alstm_pred = alstm_results['model'].predict([X_returns_test, X_chars_test])
    
    print("Training FactorVAE model...")
    factorvae_results = BaselineModels.factorVAE(X_returns_train, X_chars_train, y_train)
    factorvae_pred = factorvae_results['model'].predict([X_returns_test, X_chars_test])
    
    # Prepare results dictionary
    models_results = {
        'RVRAE': {
            'y_pred': rvrae_pred,
            'metrics': rvrae_metrics,
            'portfolio': portfolio_simulation(y_test, rvrae_pred),
            'portfolio_with_costs': portfolio_simulation(y_test, rvrae_pred, transaction_cost=0.0030),
            'rank_ic': rank_ic(y_test, rvrae_pred)
        },
        'IPCA': {
            'y_pred': ipca_results['y_pred'],
            'metrics': {
                'mse': ipca_results['mse'],
                'r2': ipca_results['r2'],
                'total_r2': ipca_results['total_r2'],
                'predictive_r2': ipca_results['predictive_r2']
            },
            'portfolio': portfolio_simulation(y_test, ipca_results['y_pred']),
            'portfolio_with_costs': portfolio_simulation(y_test, ipca_results['y_pred'], transaction_cost=0.0030),
            'rank_ic': rank_ic(y_test, ipca_results['y_pred'])
        },
        'ALSTM': {
            'y_pred': alstm_pred,
            'metrics': {
                'mse': alstm_results['mse'],
                'r2': alstm_results['r2'],
                'total_r2': alstm_results['total_r2'],
                'predictive_r2': alstm_results['predictive_r2']
            },
            'portfolio': portfolio_simulation(y_test, alstm_pred),
            'portfolio_with_costs': portfolio_simulation(y_test, alstm_pred, transaction_cost=0.0030),
            'rank_ic': rank_ic(y_test, alstm_pred)
        },
        'FactorVAE': {
            'y_pred': factorvae_pred,
            'metrics': {
                'mse': factorvae_results['mse'],
                'r2': factorvae_results['r2'],
                'total_r2': factorvae_results['total_r2'],
                'predictive_r2': factorvae_results['predictive_r2']
            },
            'portfolio': portfolio_simulation(y_test, factorvae_pred),
            'portfolio_with_costs': portfolio_simulation(y_test, factorvae_pred, transaction_cost=0.0030),
            'rank_ic': rank_ic(y_test, factorvae_pred)
        }
    }
    
    # Evaluate model performance
    print("\nModel Performance Summary:")
    summary = evaluate_model_performance(models_results)
    print(summary)
    
    # Test robustness with missing stocks
    print("\nTesting robustness with missing stocks...")
    
    robustness_results = {}
    missing_sizes = [50, 100, 150]
    
    for m in missing_sizes:
        print(f"Testing with {m} missing stocks...")
        
        # Generate new data with more stocks
        n_stocks = 50 + m
        robust_data = generate_synthetic_data(n_samples=500, n_stocks=n_stocks, n_chars=46, n_timesteps=12, n_factors=5, seed=43)
        
        # Split data
        X_returns_train_robust = robust_data['historical_returns'][:train_size, :, :50]
        X_chars_train_robust = robust_data['chars'][:train_size]
        y_train_robust = robust_data['returns'][:train_size, :50]
        
        X_returns_test_robust = robust_data['historical_returns'][train_size + val_size:, :, :]
        X_chars_test_robust = robust_data['chars'][train_size + val_size:]
        y_test_robust = robust_data['returns'][train_size + val_size:, :]
        
        # Initialize and train RVRAE model
        rvrae_robust = RVRAE(
            input_dim=50,
            char_dim=X_chars_train_robust.shape[2],
            latent_dim=5,
            timesteps=X_returns_train_robust.shape[1]
        )
        rvrae_robust.build_model()
        rvrae_robust.compile_model()
        
        rvrae_robust.fit(
            X_returns_train_robust, X_chars_train_robust, y_train_robust,
            epochs=30,
            batch_size=32,
            verbose=0
        )
        
        # Predict for all stocks (including missing ones)
        # Here we need to pad the input to match the expected shape
        padded_returns = np.zeros((X_returns_test_robust.shape[0], X_returns_test_robust.shape[1], 50))
        padded_returns[:, :, :] = X_returns_test_robust[:, :, :50]
        
        rvrae_pred_robust = rvrae_robust.predict(padded_returns, X_chars_test_robust)
        
        # Calculate Rank IC for missing stocks only
        missing_rank_ic = rank_ic(y_test_robust[:, 50:], rvrae_pred_robust[:, 50:])
        
        # Calculate Rank ICIR
        rank_icir = missing_rank_ic / np.std(rvrae_pred_robust[:, 50:].flatten())
        
        robustness_results[m] = {
            'rank_ic': missing_rank_ic,
            'rank_icir': rank_icir
        }
    
    # Display robustness results
    print("\nRobustness Results (RVRAE):")
    for m in missing_sizes:
        print(f"Missing stocks: {m}")
        print(f"  Rank IC: {robustness_results[m]['rank_ic']:.4f}")
        print(f"  Rank ICIR: {robustness_results[m]['rank_icir']:.4f}")
    
    # Analyze latent factors
    print("\nAnalyzing latent factors...")
    
    # Get latent factors from RVRAE
    rvrae_factors = rvrae.get_latent_factors(X_returns_test)
    
    # Compute correlation with true factors
    true_factors = data['factors'][train_size + val_size:]
    
    # We need to align factors (they may be permuted)
    correlation_matrix = np.abs(np.corrcoef(rvrae_factors.T, true_factors.T)[:5, 5:])
    max_corrs = np.max(correlation_matrix, axis=1)
    
    print(f"Average correlation with true factors: {np.mean(max_corrs):.4f}")
    
    # Visualize factor distributions
    plt.figure(figsize=(15, 5))
    
    for i in range(min(5, rvrae_factors.shape[1])):
        plt.subplot(1, 5, i+1)
        sns.kdeplot(rvrae_factors[:, i], label='RVRAE')
        sns.kdeplot(true_factors[:, i], label='True')
        plt.title(f'Factor {i+1}')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Test with random missing data (robustness test)
    print("\nTesting with random missing data...")
    
    # Create a copy of the test data with some random missing values
    X_returns_test_missing = X_returns_test.copy()
    X_chars_test_missing = X_chars_test.copy()
    
    # Randomly mask 10% of the data
    mask_returns = np.random.rand(*X_returns_test.shape) < 0.1
    mask_chars = np.random.rand(*X_chars_test.shape) < 0.1
    
    # Replace masked values with zeros
    X_returns_test_missing[mask_returns] = 0
    X_chars_test_missing[mask_chars] = 0
    
    # Predict with missing data
    rvrae_pred_missing = rvrae.predict(X_returns_test_missing, X_chars_test_missing)
    
    # Calculate metrics
    rvrae_metrics_missing = {
        'mse': mean_squared_error(y_test, rvrae_pred_missing),
        'r2': r2_score(y_test, rvrae_pred_missing),
        'total_r2': 1 - np.sum((y_test - rvrae_pred_missing)**2) / np.sum(y_test**2),
        'predictive_r2': 1 - np.sum((y_test - rvrae_pred_missing)**2) / np.sum((y_test - np.mean(y_test, axis=0))**2)
    }
    
    print("RVRAE performance with missing data:")
    print(f"  MSE: {rvrae_metrics_missing['mse']:.6f}")
    print(f"  Total R2: {rvrae_metrics_missing['total_r2']*100:.2f}%")
    print(f"  Predictive R2: {rvrae_metrics_missing['predictive_r2']*100:.2f}%")
    
    # Compare with original performance
    print("\nComparison with original performance:")
    print(f"  Total R2 decrease: {(rvrae_metrics['total_r2'] - rvrae_metrics_missing['total_r2'])*100:.2f}%")
    print(f"  Predictive R2 decrease: {(rvrae_metrics['predictive_r2'] - rvrae_metrics_missing['predictive_r2'])*100:.2f}%")


if __name__ == "__main__":
    main()