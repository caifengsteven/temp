import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import yfinance as yf
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class StockDataGenerator:
    def __init__(self, stock_symbols, start_date, end_date, input_length=20, output_length=20):
        """
        Initialize the stock data generator.
        
        Parameters:
        -----------
        stock_symbols : list
            List of stock symbols to download data for
        start_date : str
            Start date for data in the format 'YYYY-MM-DD'
        end_date : str
            End date for data in the format 'YYYY-MM-DD'
        input_length : int
            Number of trading days in each input frame
        output_length : int
            Number of trading days to predict trends for
        """
        self.stock_symbols = stock_symbols
        self.start_date = start_date
        self.end_date = end_date
        self.input_length = input_length
        self.output_length = output_length
        self.data = {}
        
    def download_data(self):
        """Download historical price data for each stock symbol."""
        for symbol in self.stock_symbols:
            data = yf.download(symbol, start=self.start_date, end=self.end_date)
            self.data[symbol] = data
        
    def generate_dataset(self, train_ratio=0.65, val_ratio=0.1, test_ratio=0.25):
        """
        Generate dataset for training, validation, and testing.
        
        Parameters:
        -----------
        train_ratio : float
            Ratio of data for training
        val_ratio : float
            Ratio of data for validation
        test_ratio : float
            Ratio of data for testing
        
        Returns:
        --------
        datasets : dict
            Dictionary containing training, validation, and testing datasets
        """
        datasets = {'train': [], 'val': [], 'test': []}
        
        for symbol in self.stock_symbols:
            data = self.data[symbol]
            
            # Extract price features
            price_data = data[['Open', 'High', 'Low', 'Close']].values
            
            # Scale the data to [0, 1] range
            scaled_data = self._scale_data(price_data)
            
            # Generate input frames and labels
            input_frames, labels = self._generate_frames_and_labels(scaled_data)
            
            # Split data into training, validation, and testing sets
            n_samples = len(input_frames)
            train_size = int(n_samples * train_ratio)
            val_size = int(n_samples * val_ratio)
            
            datasets['train'].extend(list(zip(input_frames[:train_size], labels[:train_size])))
            datasets['val'].extend(list(zip(input_frames[train_size:train_size+val_size], 
                                         labels[train_size:train_size+val_size])))
            datasets['test'].extend(list(zip(input_frames[train_size+val_size:], 
                                          labels[train_size+val_size:])))
        
        # Convert to numpy arrays
        for split in datasets:
            if datasets[split]:
                inputs, outputs = zip(*datasets[split])
                datasets[split] = (np.array(inputs), np.array(outputs))
            else:
                datasets[split] = (np.array([]), np.array([]))
        
        return datasets
    
    def _scale_data(self, data):
        """
        Scale data to the range [0, 1].
        
        Parameters:
        -----------
        data : ndarray
            Array of price data
        
        Returns:
        --------
        scaled_data : ndarray
            Scaled price data
        """
        min_val = np.min(data)
        max_val = np.max(data)
        
        scaled_data = (data - min_val) / (max_val - min_val)
        
        return scaled_data
    
    def _generate_frames_and_labels(self, scaled_data):
        """
        Generate input frames and labels from scaled data.
        
        Parameters:
        -----------
        scaled_data : ndarray
            Scaled price data
        
        Returns:
        --------
        input_frames : list
            List of input frames
        labels : list
            List of labels (segmentation masks)
        """
        input_frames = []
        labels = []
        
        for i in range(len(scaled_data) - self.input_length - self.output_length + 1):
            # Extract input frame
            input_frame = scaled_data[i:i+self.input_length]
            
            # Extract output frame
            output_frame = scaled_data[i+self.input_length:i+self.input_length+self.output_length]
            
            # Create label (segmentation mask)
            label = np.zeros((self.output_length, 4))
            for j in range(self.output_length):
                for k in range(4):  # Open, High, Low, Close
                    # Compare with the last day of the input frame
                    if output_frame[j, k] > input_frame[-1, k]:
                        label[j, k] = 1
            
            input_frames.append(input_frame)
            labels.append(label)
        
        return input_frames, labels

class StockTrendModel:
    def __init__(self, input_shape, output_shape, n_frames=1):
        """
        Initialize the stock trend prediction model.
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of each input frame (days, features)
        output_shape : tuple
            Shape of each output frame (days, features)
        n_frames : int
            Number of input frames to use in parallel
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_frames = n_frames
        
    def build_model(self):
        """
        Build the semantic segmentation model for stock trend prediction.
        
        Returns:
        --------
        model : tf.keras.Model
            The built model
        """
        # Input layers for each frame
        inputs = []
        for i in range(self.n_frames):
            inputs.append(layers.Input(shape=self.input_shape))
        
        # Encoder streams for each input frame
        encoder_outputs = []
        for i in range(self.n_frames):
            # Reshape to add a channel dimension for 2D convolutions
            reshaped = layers.Reshape((*self.input_shape, 1))(inputs[i])
            
            # Encoder stream
            x = self._encoder_stream(reshaped)
            encoder_outputs.append(x)
        
        # Concatenate encoder outputs at each scale
        encoded_features = []
        for i in range(3):  # We have 3 scales in the encoder
            scale_features = [outputs[i] for outputs in encoder_outputs]
            if len(scale_features) > 1:
                concatenated = layers.Concatenate(axis=-1)(scale_features)
                # Apply convolution to reduce channels
                encoded_features.append(layers.Conv2D(64, (1, 1), padding='same')(concatenated))
            else:
                encoded_features.append(scale_features[0])
        
        # Decoder
        decoded = self._decoder(encoded_features)
        
        # Output layer
        output_shape_with_channel = (*self.output_shape, 1)
        reshaped_decoded = layers.Reshape(output_shape_with_channel)(decoded)
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(reshaped_decoded)
        outputs = layers.Reshape(self.output_shape)(outputs)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def _encoder_stream(self, inputs):
        """
        Create an encoder stream with ASPP blocks.
        
        Parameters:
        -----------
        inputs : tf.Tensor
            Input tensor
        
        Returns:
        --------
        outputs : list
            List of feature maps at different scales
        """
        # First ASPP block
        x = self._aspp_block(inputs, 32)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        out1 = x
        
        # Second ASPP block
        x = self._aspp_block(x, 64)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        out2 = x
        
        # Third ASPP block
        x = self._aspp_block(x, 128)
        out3 = x
        
        return [out1, out2, out3]
    
    def _aspp_block(self, inputs, filters):
        """
        Create an Atrous Spatial Pyramid Pooling block.
        
        Parameters:
        -----------
        inputs : tf.Tensor
            Input tensor
        filters : int
            Number of filters for each branch
        
        Returns:
        --------
        x : tf.Tensor
            Output tensor
        """
        # Atrous convolutions with different dilation rates
        atrous1 = layers.Conv2D(filters, (3, 3), padding='same', dilation_rate=(1, 1))(inputs)
        atrous1 = layers.BatchNormalization()(atrous1)
        atrous1 = layers.Activation('relu')(atrous1)
        
        atrous2 = layers.Conv2D(filters, (3, 3), padding='same', dilation_rate=(2, 2))(inputs)
        atrous2 = layers.BatchNormalization()(atrous2)
        atrous2 = layers.Activation('relu')(atrous2)
        
        atrous3 = layers.Conv2D(filters, (3, 3), padding='same', dilation_rate=(3, 3))(inputs)
        atrous3 = layers.BatchNormalization()(atrous3)
        atrous3 = layers.Activation('relu')(atrous3)
        
        # Concatenate atrous convolutions
        x = layers.Concatenate()([atrous1, atrous2, atrous3])
        
        # Apply 1x1 convolution to reduce channels
        x = layers.Conv2D(filters, (1, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        return x
    
    def _decoder(self, encoded_features):
        """
        Create a decoder with skip connections.
        
        Parameters:
        -----------
        encoded_features : list
            List of feature maps at different scales
        
        Returns:
        --------
        x : tf.Tensor
            Output tensor
        """
        # Extract features at different scales
        feat1, feat2, feat3 = encoded_features
        
        # Upsample feat3 and concatenate with feat2
        x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(feat3)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Concatenate()([x, feat2])
        
        # Apply convolution
        x = layers.Conv2D(64, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Upsample and concatenate with feat1
        x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Concatenate()([x, feat1])
        
        # Apply convolution
        x = layers.Conv2D(32, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Final upsampling to match output shape
        x = layers.Conv2DTranspose(16, (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        return x
    
    def compile_model(self, model):
        """
        Compile the model with optimizer and loss function.
        
        Parameters:
        -----------
        model : tf.keras.Model
            The model to compile
        
        Returns:
        --------
        model : tf.keras.Model
            The compiled model
        """
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, model, train_data, val_data, epochs=100, batch_size=32):
        """
        Train the model.
        
        Parameters:
        -----------
        model : tf.keras.Model
            The model to train
        train_data : tuple
            Tuple of (inputs, outputs) for training
        val_data : tuple
            Tuple of (inputs, outputs) for validation
        epochs : int
            Number of epochs to train for
        batch_size : int
            Batch size for training
        
        Returns:
        --------
        history : tf.keras.callbacks.History
            Training history
        """
        # Prepare callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
        ]
        
        # Prepare inputs
        if self.n_frames == 1:
            train_inputs = train_data[0]
            val_inputs = val_data[0]
        else:
            # For multiple frames, we need to split the inputs
            train_inputs = [train_data[0][:, i*self.input_shape[0]:(i+1)*self.input_shape[0], :] 
                          for i in range(self.n_frames)]
            val_inputs = [val_data[0][:, i*self.input_shape[0]:(i+1)*self.input_shape[0], :] 
                        for i in range(self.n_frames)]
        
        # Train the model
        history = model.fit(
            train_inputs, train_data[1],
            validation_data=(val_inputs, val_data[1]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, model, test_data):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        model : tf.keras.Model
            The trained model
        test_data : tuple
            Tuple of (inputs, outputs) for testing
        
        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        # Prepare inputs
        if self.n_frames == 1:
            test_inputs = test_data[0]
        else:
            # For multiple frames, we need to split the inputs
            test_inputs = [test_data[0][:, i*self.input_shape[0]:(i+1)*self.input_shape[0], :] 
                         for i in range(self.n_frames)]
        
        # Make predictions
        y_pred = model.predict(test_inputs)
        y_true = test_data[1]
        
        # Threshold predictions to get binary values
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {}
        
        # Overall metrics
        metrics['accuracy'] = accuracy_score(y_true.flatten(), y_pred_binary.flatten())
        metrics['auc'] = roc_auc_score(y_true.flatten(), y_pred.flatten())
        metrics['precision'] = precision_score(y_true.flatten(), y_pred_binary.flatten())
        metrics['recall'] = recall_score(y_true.flatten(), y_pred_binary.flatten())
        metrics['f1'] = f1_score(y_true.flatten(), y_pred_binary.flatten())
        
        # Day-wise metrics
        day_metrics = []
        for day in range(self.output_shape[0]):
            day_true = y_true[:, day, :]
            day_pred = y_pred[:, day, :]
            day_pred_binary = (day_pred > 0.5).astype(int)
            
            day_accuracy = accuracy_score(day_true.flatten(), day_pred_binary.flatten())
            day_auc = roc_auc_score(day_true.flatten(), day_pred.flatten())
            day_precision = precision_score(day_true.flatten(), day_pred_binary.flatten())
            day_recall = recall_score(day_true.flatten(), day_pred_binary.flatten())
            day_f1 = f1_score(day_true.flatten(), day_pred_binary.flatten())
            
            day_metrics.append({
                'day': day + 1,
                'accuracy': day_accuracy,
                'auc': day_auc,
                'precision': day_precision,
                'recall': day_recall,
                'f1': day_f1
            })
        
        metrics['day_metrics'] = day_metrics
        
        return metrics, y_pred
    
    def visualize_accuracy_map(self, metrics):
        """
        Visualize the accuracy map for each day and price.
        
        Parameters:
        -----------
        metrics : dict
            Dictionary of evaluation metrics
        """
        # Extract day-wise metrics
        day_metrics = metrics['day_metrics']
        
        # Create accuracy map
        accuracy_map = np.zeros((self.output_shape[0], 4))
        for i, day_metric in enumerate(day_metrics):
            accuracy_map[i, :] = day_metric['accuracy']
        
        # Plot accuracy map
        plt.figure(figsize=(10, 6))
        sns.heatmap(accuracy_map, annot=True, cmap='viridis', fmt='.2f',
                   xticklabels=['Open', 'High', 'Low', 'Close'],
                   yticklabels=list(range(1, self.output_shape[0] + 1)))
        plt.title('Accuracy Map for Each Day and Price')
        plt.xlabel('Price')
        plt.ylabel('Day')
        plt.show()
    
    def visualize_predictions(self, test_data, y_pred, num_samples=5):
        """
        Visualize predictions for a few test samples.
        
        Parameters:
        -----------
        test_data : tuple
            Tuple of (inputs, outputs) for testing
        y_pred : ndarray
            Predicted outputs
        num_samples : int
            Number of samples to visualize
        """
        # Get a few test samples
        if self.n_frames == 1:
            test_inputs = test_data[0][:num_samples]
        else:
            test_inputs = [test_data[0][:num_samples, i*self.input_shape[0]:(i+1)*self.input_shape[0], :] 
                         for i in range(self.n_frames)]
            # For visualization, use the first frame
            test_inputs = test_inputs[0]
        
        test_outputs = test_data[1][:num_samples]
        test_preds = y_pred[:num_samples]
        
        # Visualize each sample
        for i in range(num_samples):
            plt.figure(figsize=(15, 10))
            
            # Plot input frame
            plt.subplot(3, 1, 1)
            sns.heatmap(test_inputs[i], cmap='viridis',
                       xticklabels=['Open', 'High', 'Low', 'Close'],
                       yticklabels=list(range(1, self.input_shape[0] + 1)))
            plt.title(f'Sample {i+1}: Input Frame')
            plt.xlabel('Price')
            plt.ylabel('Day')
            
            # Plot true output
            plt.subplot(3, 1, 2)
            sns.heatmap(test_outputs[i], cmap='viridis',
                       xticklabels=['Open', 'High', 'Low', 'Close'],
                       yticklabels=list(range(1, self.output_shape[0] + 1)))
            plt.title(f'Sample {i+1}: True Output')
            plt.xlabel('Price')
            plt.ylabel('Day')
            
            # Plot predicted output
            plt.subplot(3, 1, 3)
            sns.heatmap((test_preds[i] > 0.5).astype(int), cmap='viridis',
                       xticklabels=['Open', 'High', 'Low', 'Close'],
                       yticklabels=list(range(1, self.output_shape[0] + 1)))
            plt.title(f'Sample {i+1}: Predicted Output')
            plt.xlabel('Price')
            plt.ylabel('Day')
            
            plt.tight_layout()
            plt.show()

# Create a function to compare with baseline models
def compare_with_baselines(datasets, input_shape, output_shape):
    """
    Compare the proposed model with baseline models.
    
    Parameters:
    -----------
    datasets : dict
        Dictionary containing training, validation, and testing datasets
    input_shape : tuple
        Shape of each input frame (days, features)
    output_shape : tuple
        Shape of each output frame (days, features)
    
    Returns:
    --------
    results : dict
        Dictionary containing evaluation metrics for each model
    """
    results = {}
    
    # Prepare inputs and outputs
    train_inputs = datasets['train'][0]
    train_outputs = datasets['train'][1]
    val_inputs = datasets['val'][0]
    val_outputs = datasets['val'][1]
    test_inputs = datasets['test'][0]
    test_outputs = datasets['test'][1]
    
    # Flatten inputs and outputs for traditional models
    train_inputs_flat = train_inputs.reshape(train_inputs.shape[0], -1)
    val_inputs_flat = val_inputs.reshape(val_inputs.shape[0], -1)
    test_inputs_flat = test_inputs.reshape(test_inputs.shape[0], -1)
    
    # 1. MLP model
    print("Training MLP model...")
    mlp_model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(train_inputs_flat.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(np.prod(output_shape), activation='sigmoid'),
        layers.Reshape(output_shape)
    ])
    
    mlp_model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    mlp_history = mlp_model.fit(
        train_inputs_flat, train_outputs,
        validation_data=(val_inputs_flat, val_outputs),
        epochs=50,
        batch_size=32,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=0
    )
    
    # Evaluate MLP model
    mlp_preds = mlp_model.predict(test_inputs_flat)
    mlp_preds_binary = (mlp_preds > 0.5).astype(int)
    
    results['MLP'] = {
        'accuracy': accuracy_score(test_outputs.flatten(), mlp_preds_binary.flatten()),
        'auc': roc_auc_score(test_outputs.flatten(), mlp_preds.flatten()),
        'precision': precision_score(test_outputs.flatten(), mlp_preds_binary.flatten()),
        'recall': recall_score(test_outputs.flatten(), mlp_preds_binary.flatten()),
        'f1': f1_score(test_outputs.flatten(), mlp_preds_binary.flatten())
    }
    
    # 2. CNN+FC model
    print("Training CNN+FC model...")
    cnn_fc_model = models.Sequential([
        layers.Reshape((*input_shape, 1), input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(np.prod(output_shape), activation='sigmoid'),
        layers.Reshape(output_shape)
    ])
    
    cnn_fc_model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    cnn_fc_history = cnn_fc_model.fit(
        train_inputs, train_outputs,
        validation_data=(val_inputs, val_outputs),
        epochs=50,
        batch_size=32,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=0
    )
    
    # Evaluate CNN+FC model
    cnn_fc_preds = cnn_fc_model.predict(test_inputs)
    cnn_fc_preds_binary = (cnn_fc_preds > 0.5).astype(int)
    
    results['CNN+FC'] = {
        'accuracy': accuracy_score(test_outputs.flatten(), cnn_fc_preds_binary.flatten()),
        'auc': roc_auc_score(test_outputs.flatten(), cnn_fc_preds.flatten()),
        'precision': precision_score(test_outputs.flatten(), cnn_fc_preds_binary.flatten()),
        'recall': recall_score(test_outputs.flatten(), cnn_fc_preds_binary.flatten()),
        'f1': f1_score(test_outputs.flatten(), cnn_fc_preds_binary.flatten())
    }
    
    # 3. CNN+LSTM model
    print("Training CNN+LSTM model...")
    cnn_lstm_input = layers.Input(shape=input_shape)
    cnn_lstm_reshaped = layers.Reshape((*input_shape, 1))(cnn_lstm_input)
    cnn_lstm_conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(cnn_lstm_reshaped)
    cnn_lstm_pool1 = layers.MaxPooling2D(pool_size=(2, 2))(cnn_lstm_conv1)
    cnn_lstm_conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(cnn_lstm_pool1)
    cnn_lstm_pool2 = layers.MaxPooling2D(pool_size=(2, 2))(cnn_lstm_conv2)
    
    # Reshape for LSTM
    cnn_lstm_reshaped2 = layers.Reshape((-1, 64 * (input_shape[1] // 4)))(cnn_lstm_pool2)
    cnn_lstm_lstm = layers.LSTM(128)(cnn_lstm_reshaped2)
    cnn_lstm_dense = layers.Dense(np.prod(output_shape), activation='sigmoid')(cnn_lstm_lstm)
    cnn_lstm_output = layers.Reshape(output_shape)(cnn_lstm_dense)
    
    cnn_lstm_model = models.Model(inputs=cnn_lstm_input, outputs=cnn_lstm_output)
    
    cnn_lstm_model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    cnn_lstm_history = cnn_lstm_model.fit(
        train_inputs, train_outputs,
        validation_data=(val_inputs, val_outputs),
        epochs=50,
        batch_size=32,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=0
    )
    
    # Evaluate CNN+LSTM model
    cnn_lstm_preds = cnn_lstm_model.predict(test_inputs)
    cnn_lstm_preds_binary = (cnn_lstm_preds > 0.5).astype(int)
    
    results['CNN+LSTM'] = {
        'accuracy': accuracy_score(test_outputs.flatten(), cnn_lstm_preds_binary.flatten()),
        'auc': roc_auc_score(test_outputs.flatten(), cnn_lstm_preds.flatten()),
        'precision': precision_score(test_outputs.flatten(), cnn_lstm_preds_binary.flatten()),
        'recall': recall_score(test_outputs.flatten(), cnn_lstm_preds_binary.flatten()),
        'f1': f1_score(test_outputs.flatten(), cnn_lstm_preds_binary.flatten())
    }
    
    return results

def main():
    # Define stock symbols, time period, and model parameters
    stock_symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META']
    start_date = '2010-01-01'
    end_date = '2023-01-01'
    input_length = 20  # 20 trading days for input
    output_length = 20  # 20 trading days for output
    
    # Create data generator
    print("Creating data generator...")
    data_generator = StockDataGenerator(
        stock_symbols=stock_symbols,
        start_date=start_date,
        end_date=end_date,
        input_length=input_length,
        output_length=output_length
    )
    
    # Download data
    print("Downloading data...")
    data_generator.download_data()
    
    # Generate dataset
    print("Generating dataset...")
    datasets = data_generator.generate_dataset()
    
    # Print dataset sizes
    for split in datasets:
        if len(datasets[split]) > 0:
            print(f"{split} set: {datasets[split][0].shape[0]} samples")
    
    # Define model parameters
    input_shape = (input_length, 4)  # (days, features)
    output_shape = (output_length, 4)  # (days, features)
    
    # Create and train model with 1 frame
    print("\nTraining model with 1 frame...")
    stock_trend_model_1 = StockTrendModel(input_shape, output_shape, n_frames=1)
    model_1 = stock_trend_model_1.build_model()
    model_1 = stock_trend_model_1.compile_model(model_1)
    
    # Print model summary
    model_1.summary()
    
    # Train model
    history_1 = stock_trend_model_1.train_model(
        model_1,
        datasets['train'],
        datasets['val'],
        epochs=100,
        batch_size=32
    )
    
    # Evaluate model
    print("\nEvaluating model with 1 frame...")
    metrics_1, preds_1 = stock_trend_model_1.evaluate_model(model_1, datasets['test'])
    
    # Print overall metrics
    print(f"Overall Accuracy: {metrics_1['accuracy']:.4f}")
    print(f"Overall AUC: {metrics_1['auc']:.4f}")
    print(f"Overall Precision: {metrics_1['precision']:.4f}")
    print(f"Overall Recall: {metrics_1['recall']:.4f}")
    print(f"Overall F1 Score: {metrics_1['f1']:.4f}")
    
    # Print day-wise metrics for days 1, 5, 10, and 20
    for day in [0, 4, 9, 19]:
        day_metric = metrics_1['day_metrics'][day]
        print(f"\nDay {day_metric['day']} Metrics:")
        print(f"Accuracy: {day_metric['accuracy']:.4f}")
        print(f"AUC: {day_metric['auc']:.4f}")
        print(f"F1 Score: {day_metric['f1']:.4f}")
    
    # Visualize accuracy map
    stock_trend_model_1.visualize_accuracy_map(metrics_1)
    
    # Visualize predictions
    stock_trend_model_1.visualize_predictions(datasets['test'], preds_1, num_samples=3)
    
    # Create and train model with multiple frames (9 frames as per the paper)
    print("\nTraining model with 9 frames...")
    stock_trend_model_9 = StockTrendModel(input_shape, output_shape, n_frames=9)
    model_9 = stock_trend_model_9.build_model()
    model_9 = stock_trend_model_9.compile_model(model_9)
    
    # Print model summary
    model_9.summary()
    
    # Prepare datasets for multiple frames
    # For simplicity, we'll use the same data as input for all frames
    # In a real implementation, consecutive frames would be used
    multi_frame_datasets = {}
    for split in datasets:
        if len(datasets[split]) > 0:
            inputs = [datasets[split][0] for _ in range(9)]
            multi_frame_datasets[split] = (inputs, datasets[split][1])
    
    # Train model
    history_9 = stock_trend_model_9.train_model(
        model_9,
        multi_frame_datasets['train'],
        multi_frame_datasets['val'],
        epochs=100,
        batch_size=32
    )
    
    # Evaluate model
    print("\nEvaluating model with 9 frames...")
    metrics_9, preds_9 = stock_trend_model_9.evaluate_model(model_9, multi_frame_datasets['test'])
    
    # Print overall metrics
    print(f"Overall Accuracy: {metrics_9['accuracy']:.4f}")
    print(f"Overall AUC: {metrics_9['auc']:.4f}")
    print(f"Overall Precision: {metrics_9['precision']:.4f}")
    print(f"Overall Recall: {metrics_9['recall']:.4f}")
    print(f"Overall F1 Score: {metrics_9['f1']:.4f}")
    
    # Print day-wise metrics for days 1, 5, 10, and 20
    for day in [0, 4, 9, 19]:
        day_metric = metrics_9['day_metrics'][day]
        print(f"\nDay {day_metric['day']} Metrics:")
        print(f"Accuracy: {day_metric['accuracy']:.4f}")
        print(f"AUC: {day_metric['auc']:.4f}")
        print(f"F1 Score: {day_metric['f1']:.4f}")
    
    # Visualize accuracy map
    stock_trend_model_9.visualize_accuracy_map(metrics_9)
    
    # Compare with baseline models
    print("\nComparing with baseline models...")
    baseline_results = compare_with_baselines(datasets, input_shape, output_shape)
    
    # Print comparison results
    print("\nModel Comparison:")
    print(f"{'Model':<10} {'Accuracy':<10} {'AUC':<10} {'F1 Score':<10}")
    print("-" * 40)
    
    print(f"{'Proposed-1':<10} {metrics_1['accuracy']:.4f} {metrics_1['auc']:.4f} {metrics_1['f1']:.4f}")
    print(f"{'Proposed-9':<10} {metrics_9['accuracy']:.4f} {metrics_9['auc']:.4f} {metrics_9['f1']:.4f}")
    
    for model_name, metrics in baseline_results.items():
        print(f"{model_name:<10} {metrics['accuracy']:.4f} {metrics['auc']:.4f} {metrics['f1']:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_1.history['loss'], label='1-Frame Train')
    plt.plot(history_1.history['val_loss'], label='1-Frame Val')
    plt.plot(history_9.history['loss'], label='9-Frame Train')
    plt.plot(history_9.history['val_loss'], label='9-Frame Val')
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history_1.history['accuracy'], label='1-Frame Train')
    plt.plot(history_1.history['val_accuracy'], label='1-Frame Val')
    plt.plot(history_9.history['accuracy'], label='9-Frame Train')
    plt.plot(history_9.history['val_accuracy'], label='9-Frame Val')
    plt.title('Accuracy During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Trading strategy based on predictions
    print("\nSimulating trading strategy...")
    
    # Function to simulate trading strategy
    def simulate_trading_strategy(predictions, test_data, initial_capital=10000, commission=0.001):
        # Get test inputs and true outputs
        test_inputs = test_data[0]
        test_outputs = test_data[1]
        
        # Get the last day's prices from test inputs
        if isinstance(test_inputs, list):  # For multi-frame model
            last_day_prices = test_inputs[0][:, -1, :]  # Last day of first frame
        else:
            last_day_prices = test_inputs[:, -1, :]  # Last day of input frame
        
        # Initialize portfolio
        portfolio_value = np.zeros(len(predictions) + 1)
        portfolio_value[0] = initial_capital
        
        # Trading strategy: Buy if predicted trend is up, sell if down
        for i in range(len(predictions)):
            # Get prediction for next day's trends
            next_day_pred = predictions[i, 0, :]  # First day prediction
            
            # Get the prices for the current day
            current_prices = last_day_prices[i]
            
            # Initialize position
            position = np.zeros(4)  # For Open, High, Low, Close
            
            # Decide position based on predictions
            for j in range(4):
                if next_day_pred[j] > 0.5:  # Predicted up trend
                    position[j] = 1  # Long position
                else:
                    position[j] = -1  # Short position
            
            # Calculate returns
            actual_trends = test_outputs[i, 0, :]
            returns = np.zeros(4)
            
            for j in range(4):
                if actual_trends[j] > 0.5:  # Actually went up
                    returns[j] = 0.01  # Assume 1% return for simplicity
                else:
                    returns[j] = -0.01  # Assume -1% return for simplicity
            
            # Calculate portfolio return
            portfolio_return = np.sum(position * returns)
            
            # Apply commission
            commission_cost = np.sum(np.abs(position)) * commission
            
            # Update portfolio value
            portfolio_value[i+1] = portfolio_value[i] * (1 + portfolio_return - commission_cost)
        
        return portfolio_value
    
    # Simulate trading strategies
    portfolio_1 = simulate_trading_strategy(preds_1, datasets['test'])
    portfolio_9 = simulate_trading_strategy(preds_9, multi_frame_datasets['test'])
    
    # Plot portfolio values
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_1, label='1-Frame Model')
    plt.plot(portfolio_9, label='9-Frame Model')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Trading Day')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Calculate trading metrics
    def calculate_trading_metrics(portfolio_values):
        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Calculate metrics
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        max_drawdown = np.min(portfolio_values / np.maximum.accumulate(portfolio_values)) - 1
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    # Calculate trading metrics
    metrics_1_trading = calculate_trading_metrics(portfolio_1)
    metrics_9_trading = calculate_trading_metrics(portfolio_9)
    
    # Print trading metrics
    print("\nTrading Metrics:")
    print(f"{'Model':<10} {'Total Return':<15} {'Annual Return':<15} {'Sharpe Ratio':<15} {'Max Drawdown':<15}")
    print("-" * 70)
    
    print(f"{'1-Frame':<10} {metrics_1_trading['total_return']:.4f} {metrics_1_trading['annualized_return']:.4f} {metrics_1_trading['sharpe_ratio']:.4f} {metrics_1_trading['max_drawdown']:.4f}")
    print(f"{'9-Frame':<10} {metrics_9_trading['total_return']:.4f} {metrics_9_trading['annualized_return']:.4f} {metrics_9_trading['sharpe_ratio']:.4f} {metrics_9_trading['max_drawdown']:.4f}")

if __name__ == "__main__":
    main()