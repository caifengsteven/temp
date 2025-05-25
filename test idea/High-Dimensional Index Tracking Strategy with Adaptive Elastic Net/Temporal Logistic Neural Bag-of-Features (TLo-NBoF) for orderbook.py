import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, f1_score, precision_recall_fscore_support
import time
import datetime as dt
import random
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class LogisticNBoFLayer(nn.Module):
    """
    Logistic Neural Bag-of-Features layer
    """
    def __init__(self, feature_dim, num_codewords=256, use_kernel_param_learning=True, 
                 use_adaptive_scaling=True):
        super(LogisticNBoFLayer, self).__init__()
        self.feature_dim = feature_dim
        self.num_codewords = num_codewords
        self.use_kernel_param_learning = use_kernel_param_learning
        self.use_adaptive_scaling = use_adaptive_scaling
        
        # Initialize codebook randomly (trainable parameters)
        self.codebook = nn.Parameter(torch.randn(num_codewords, feature_dim))
        
        # Initialize kernel parameters (as in the paper)
        self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=use_kernel_param_learning)
        self.beta = nn.Parameter(torch.tensor(0.0), requires_grad=use_kernel_param_learning)
        
        # Initialize scaling parameters
        self.cs = nn.Parameter(torch.tensor(float(num_codewords)), requires_grad=use_adaptive_scaling)
        self.cu = nn.Parameter(torch.tensor(15.0), requires_grad=use_adaptive_scaling)  # Default time_steps
    
    def forward(self, x):
        # x shape: [batch_size, time_steps, feature_dim]
        batch_size, time_steps, _ = x.shape
        
        # Make sure the tensor is contiguous before reshaping
        x_reshaped = x.contiguous().view(batch_size * time_steps, self.feature_dim)
        
        # Compute logistic kernel (equation 7 in the paper)
        # Similarity between each feature vector and each codeword
        # dot product: x_reshaped @ codebook.T
        similarity = 2 * self.alpha * (x_reshaped @ self.codebook.t()) + 2 * self.beta
        similarity = torch.sigmoid(similarity)  # shape: [batch_size * time_steps, num_codewords]
        
        # Normalize similarities (equation 9 in the paper)
        sum_similarities = similarity.sum(dim=1, keepdim=True)
        membership = similarity / (sum_similarities + 1e-10)  # Add epsilon for numerical stability
        
        # Apply adaptive scaling for membership vectors
        if self.use_adaptive_scaling:
            membership = membership * self.cu
        
        # Reshape back to original batch structure
        membership = membership.view(batch_size, time_steps, self.num_codewords)
        
        # Calculate histogram (equation 8 in the paper) - average over time dimension
        histogram = membership.mean(dim=1)  # shape: [batch_size, num_codewords]
        
        # Apply adaptive scaling for histogram
        if self.use_adaptive_scaling:
            histogram = histogram * self.cs
        
        return histogram


class TemporalLoNBoF(nn.Module):
    """
    Temporal Logistic Neural Bag-of-Features model
    """
    def __init__(self, input_shape=(15, 144), num_codewords=256, num_temporal_regions=3, 
                 num_filters=256, kernel_size=5, use_deep_features=True, 
                 use_temporal_modeling=True, use_kernel_param_learning=True,
                 use_adaptive_scaling=True, dropout_rate=0.3):
        super(TemporalLoNBoF, self).__init__()
        self.input_shape = input_shape
        self.num_codewords = num_codewords
        self.num_temporal_regions = num_temporal_regions
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.use_deep_features = use_deep_features
        self.use_temporal_modeling = use_temporal_modeling
        self.use_kernel_param_learning = use_kernel_param_learning
        self.use_adaptive_scaling = use_adaptive_scaling
        self.dropout_rate = dropout_rate
        
        # Feature extraction layer
        if self.use_deep_features:
            self.feature_extractor = nn.Sequential(
                nn.Conv1d(in_channels=input_shape[1], out_channels=num_filters, 
                          kernel_size=kernel_size, padding=kernel_size//2),
                nn.ReLU()
            )
            feature_dim = num_filters
        else:
            feature_dim = input_shape[1]
        
        # Temporal BoF layers
        if self.use_temporal_modeling:
            self.region_size = input_shape[0] // num_temporal_regions
            self.bof_layers = nn.ModuleList([
                LogisticNBoFLayer(
                    feature_dim=feature_dim, 
                    num_codewords=num_codewords,
                    use_kernel_param_learning=use_kernel_param_learning,
                    use_adaptive_scaling=use_adaptive_scaling
                ) for _ in range(num_temporal_regions)
            ])
            histogram_dim = num_codewords * num_temporal_regions
        else:
            self.bof_layers = LogisticNBoFLayer(
                feature_dim=feature_dim,
                num_codewords=num_codewords,
                use_kernel_param_learning=use_kernel_param_learning,
                use_adaptive_scaling=use_adaptive_scaling
            )
            histogram_dim = num_codewords
        
        # Fully connected layers for classification
        self.classifier = nn.Sequential(
            nn.Linear(histogram_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 3)  # 3 classes: up, stationary, down
        )
    
    def forward(self, x):
        # x shape: [batch_size, time_steps, features]
        batch_size = x.shape[0]
        
        # Apply feature extraction if enabled
        if self.use_deep_features:
            # Transpose for Conv1D (expects [batch, channels, time])
            x = x.transpose(1, 2)
            x = self.feature_extractor(x)
            # Transpose back to [batch, time, features]
            x = x.transpose(1, 2)
        
        # Apply BoF layer(s)
        if self.use_temporal_modeling:
            histograms = []
            
            for i in range(self.num_temporal_regions):
                # Extract temporal region
                start_idx = i * self.region_size
                end_idx = (i + 1) * self.region_size if i < self.num_temporal_regions - 1 else self.input_shape[0]
                region = x[:, start_idx:end_idx, :]
                
                # Apply BoF to the region
                hist = self.bof_layers[i](region)
                histograms.append(hist)
            
            # Concatenate histograms from all temporal regions
            x = torch.cat(histograms, dim=1)
        else:
            x = self.bof_layers(x)
        
        # Apply classifier
        x = self.classifier(x)
        
        return x


class LOBDataset(Dataset):
    """
    Dataset for Limit Order Book data
    """
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class LOBDataProcessor:
    """
    Process limit order book (LOB) data for the TLo-NBoF model
    """
    
    def __init__(self, time_steps=15, prediction_horizon=10, price_change_threshold=0.0001):
        """
        Initialize the LOB data processor
        
        Parameters:
        -----------
        time_steps : int
            Number of time steps to include in each sample
        prediction_horizon : int
            Number of time steps ahead to predict
        price_change_threshold : float
            Threshold for classifying price changes as up/down/stationary
        """
        self.time_steps = time_steps
        self.prediction_horizon = prediction_horizon
        self.price_change_threshold = price_change_threshold
        self.scaler = StandardScaler()
        
    def process_lob_data(self, data, is_training=True):
        """
        Process limit order book data
        
        Parameters:
        -----------
        data : pandas DataFrame
            Limit order book data
        is_training : bool
            Whether this is training data (for fitting the scaler)
            
        Returns:
        --------
        X : numpy array
            Processed features
        y : numpy array
            Labels (price direction)
        """
        # Extract features
        features = self._extract_lob_features(data)
        
        # Scale features
        if is_training:
            features_scaled = self.scaler.fit_transform(features)
        else:
            features_scaled = self.scaler.transform(features)
        
        # Create time series samples
        X, y = self._create_time_series_samples(features_scaled, data)
        
        return X, y
    
    def _extract_lob_features(self, data):
        """
        Extract features from limit order book data
        
        Parameters:
        -----------
        data : pandas DataFrame
            Limit order book data
            
        Returns:
        --------
        features : numpy array
            Extracted features
        """
        # Extract basic features
        features = []
        
        # Price and volume features
        if 'ask_price_1' in data.columns and 'bid_price_1' in data.columns:
            # Mid price
            mid_price = (data['ask_price_1'] + data['bid_price_1']) / 2
            
            # Price spreads
            price_spread = data['ask_price_1'] - data['bid_price_1']
            
            # Price differences
            if 'ask_price_2' in data.columns:
                ask_diff = data['ask_price_2'] - data['ask_price_1']
                bid_diff = data['bid_price_1'] - data['bid_price_2']
                features.extend([ask_diff, bid_diff])
            
            # Volume imbalance
            if 'ask_size_1' in data.columns and 'bid_size_1' in data.columns:
                volume_imbalance = (data['bid_size_1'] - data['ask_size_1']) / (data['bid_size_1'] + data['ask_size_1'])
                
                features.extend([mid_price, price_spread, volume_imbalance])
        
        # If we have all LOB levels (as in the paper)
        lob_levels = 10  # Paper uses 10 levels
        
        # Add all available price and volume features from LOB
        for level in range(1, lob_levels + 1):
            ask_price_col = f'ask_price_{level}'
            bid_price_col = f'bid_price_{level}'
            ask_size_col = f'ask_size_{level}'
            bid_size_col = f'bid_size_{level}'
            
            if all(col in data.columns for col in [ask_price_col, bid_price_col, ask_size_col, bid_size_col]):
                features.extend([
                    data[ask_price_col], data[bid_price_col],
                    data[ask_size_col], data[bid_size_col]
                ])
        
        # If features list is empty, use all numerical columns from data
        if not features:
            features = data.select_dtypes(include=['number']).values
        else:
            features = np.column_stack(features)
            
        return features
    
    def _create_time_series_samples(self, features, data):
        """
        Create time series samples from features
        
        Parameters:
        -----------
        features : numpy array
            Scaled features
        data : pandas DataFrame
            Original data (for calculating labels)
            
        Returns:
        --------
        X : numpy array
            Time series samples
        y : numpy array
            Labels
        """
        X = []
        y = []
        
        # Calculate mid prices for labels
        if 'ask_price_1' in data.columns and 'bid_price_1' in data.columns:
            mid_prices = (data['ask_price_1'] + data['bid_price_1']) / 2
        else:
            # Use first numerical column as proxy if no price data available
            mid_prices = data.select_dtypes(include=['number']).iloc[:, 0]
        
        # Create samples
        for i in range(self.time_steps, len(features) - self.prediction_horizon):
            # Create sample
            sample = features[i - self.time_steps:i]
            X.append(sample)
            
            # Create label based on price movement
            current_price = mid_prices.iloc[i]
            future_price = mid_prices.iloc[i + self.prediction_horizon]
            
            price_change = (future_price - current_price) / current_price
            
            if price_change > self.price_change_threshold:
                # Price goes up
                label = 0
            elif price_change < -self.price_change_threshold:
                # Price goes down
                label = 2
            else:
                # Price remains stationary
                label = 1
                
            y.append(label)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        return X, y


def generate_simulated_lob_data(num_days=10, events_per_day=50000, lob_levels=10):
    """
    Generate simulated limit order book data
    
    Parameters:
    -----------
    num_days : int
        Number of days to simulate
    events_per_day : int
        Number of events per day
    lob_levels : int
        Number of price levels in the LOB
        
    Returns:
    --------
    data : dict
        Dictionary containing simulated LOB data for each day
    """
    # Initialize dictionary to store data for each day
    data = {}
    
    # Base price and volatility parameters
    base_price = 100.0
    daily_volatility = 0.015  # 1.5% daily volatility
    intraday_volatility = daily_volatility / np.sqrt(events_per_day)
    
    # Tick size
    tick_size = 0.01
    
    # Generate data for each day
    for day in range(num_days):
        # Initialize DataFrame
        columns = []
        for level in range(1, lob_levels + 1):
            columns.extend([
                f'ask_price_{level}', f'ask_size_{level}',
                f'bid_price_{level}', f'bid_size_{level}'
            ])
        
        day_data = pd.DataFrame(columns=columns)
        
        # Initial mid price with some random drift from previous day
        if day > 0:
            prev_mid_price = (data[day-1]['ask_price_1'] + data[day-1]['bid_price_1']).iloc[-1] / 2
            daily_return = np.random.normal(0, daily_volatility)
            mid_price = prev_mid_price * (1 + daily_return)
        else:
            mid_price = base_price
        
        # Generate LOB states
        for event in range(events_per_day):
            # Update mid price with random walk
            price_change = np.random.normal(0, intraday_volatility)
            mid_price *= (1 + price_change)
            
            # Generate bid and ask prices around mid price
            # Create a realistic spread that's wider during volatile periods
            spread_factor = 1.0 + abs(price_change) * 10
            spread = max(tick_size, tick_size * spread_factor)
            
            ask_price_1 = mid_price + spread / 2
            bid_price_1 = mid_price - spread / 2
            
            # Ensure bid < ask
            bid_price_1 = min(bid_price_1, ask_price_1 - tick_size)
            
            # Round to tick size
            ask_price_1 = round(ask_price_1 / tick_size) * tick_size
            bid_price_1 = round(bid_price_1 / tick_size) * tick_size
            
            # Generate remaining LOB prices
            ask_prices = [ask_price_1]
            bid_prices = [bid_price_1]
            
            for level in range(2, lob_levels + 1):
                # Add increasing spreads for higher levels
                # With some randomness for realistic variation
                next_ask = ask_prices[-1] + tick_size * (1 + 0.5 * np.random.rand())
                next_bid = bid_prices[-1] - tick_size * (1 + 0.5 * np.random.rand())
                
                ask_prices.append(next_ask)
                bid_prices.append(next_bid)
            
            # Generate order sizes
            # Typically larger at the best price, decreasing for worse prices
            ask_sizes = []
            bid_sizes = []
            
            for level in range(1, lob_levels + 1):
                # Base size with randomness, decreasing with level
                size_factor = np.exp(-0.3 * (level - 1))
                base_size = 100 * size_factor
                
                # Add randomness
                ask_size = max(1, int(base_size * (1 + 0.5 * np.random.rand())))
                bid_size = max(1, int(base_size * (1 + 0.5 * np.random.rand())))
                
                ask_sizes.append(ask_size)
                bid_sizes.append(bid_size)
            
            # Combine data for this event
            event_data = {}
            for level in range(1, lob_levels + 1):
                event_data[f'ask_price_{level}'] = ask_prices[level-1]
                event_data[f'ask_size_{level}'] = ask_sizes[level-1]
                event_data[f'bid_price_{level}'] = bid_prices[level-1]
                event_data[f'bid_size_{level}'] = bid_sizes[level-1]
            
            day_data = pd.concat([day_data, pd.DataFrame([event_data])], ignore_index=True)
        
        data[day] = day_data
    
    return data


def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=1e-4, weight_decay=1e-4):
    """
    Train the TLo-NBoF model
    
    Parameters:
    -----------
    model : TemporalLoNBoF
        Model to train
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    num_epochs : int
        Number of epochs to train
    learning_rate : float
        Learning rate for optimizer
    weight_decay : float
        Weight decay for regularization
        
    Returns:
    --------
    model : TemporalLoNBoF
        Trained model
    history : dict
        Training history
    """
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Initialize loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize lists to store metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Initialize best validation loss
    best_val_loss = float('inf')
    best_model_state = None
    
    # Training loop
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        
        # Initialize metrics
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Training
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Move data to device
            data, targets = data.to(device).float(), targets.to(device).long()
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            # Print progress
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Calculate average training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100.0 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                # Move data to device
                data, targets = data.to(device).float(), targets.to(device).long()
                
                # Forward pass
                outputs = model(data)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Update metrics
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        # Calculate average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100.0 * val_correct / val_total
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_accuracy)
        
        # Print epoch summary
        print(f'Epoch: {epoch+1}/{num_epochs}, '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            print(f'Best model saved with val loss: {best_val_loss:.4f}')
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


def evaluate_model(model, test_loader):
    """
    Evaluate the TLo-NBoF model
    
    Parameters:
    -----------
    model : TemporalLoNBoF
        Model to evaluate
    test_loader : DataLoader
        Test data loader
        
    Returns:
    --------
    metrics : dict
        Evaluation metrics
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize variables
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    criterion = nn.CrossEntropyLoss()
    
    # Initialize lists to store predictions and targets
    all_predictions = []
    all_targets = []
    
    # Evaluation loop
    with torch.no_grad():
        for data, targets in test_loader:
            # Move data to device
            data, targets = data.to(device).float(), targets.to(device).long()
            
            # Forward pass
            outputs = model(data)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Update metrics
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += targets.size(0)
            test_correct += (predicted == targets).sum().item()
            
            # Save predictions and targets
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate average metrics
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100.0 * test_correct / test_total
    
    # Calculate additional metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='macro'
    )
    
    # Calculate Cohen's Kappa
    kappa = cohen_kappa_score(all_targets, all_predictions)
    
    # Create confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Print classification report
    print(classification_report(all_targets, all_predictions))
    
    # Compile metrics
    metrics = {
        'test_loss': avg_test_loss,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'kappa': kappa,
        'confusion_matrix': cm,
        'predictions': np.array(all_predictions),
        'targets': np.array(all_targets)
    }
    
    # Print summary
    print(f'Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
    print(f'Cohen\'s Kappa: {kappa:.4f}')
    
    return metrics


def plot_training_history(history):
    """
    Plot training history
    
    Parameters:
    -----------
    history : dict
        Training history
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


def plot_confusion_matrix(cm, classes=['Up', 'Stationary', 'Down']):
    """
    Plot confusion matrix
    
    Parameters:
    -----------
    cm : numpy array
        Confusion matrix
    classes : list
        Class names
    """
    plt.figure(figsize=(8, 6))
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f'{cm[i, j]} ({cm_norm[i, j]:.2f})',
                    horizontalalignment="center",
                    color="white" if cm_norm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.show()


def implement_trading_strategy(model, test_loader, metrics):
    """
    Implement a simple trading strategy based on model predictions
    
    Parameters:
    -----------
    model : TemporalLoNBoF
        Trained model
    test_loader : DataLoader
        Test data loader
    metrics : dict
        Evaluation metrics from the model
        
    Returns:
    --------
    returns : dict
        Strategy returns
    """
    # Initialize variables
    initial_capital = 10000.0
    position = 0  # -1: short, 0: neutral, 1: long
    capital = initial_capital
    trade_history = []
    
    # Get predictions and targets
    predictions = metrics['predictions']
    targets = metrics['targets']
    
    # Get mid prices from test data (for simplicity, we use targets to determine price changes)
    # In a real implementation, you would extract actual price data
    
    # Simple implementation: use 0=up, 1=stationary, 2=down predictions
    # to determine trading decisions
    
    idx = 0
    for data, _ in test_loader:
        # Skip if we've reached the end of our predictions
        if idx >= len(predictions):
            break
        
        # Get prediction for current sample
        prediction = predictions[idx]
        
        # Make trading decision
        if prediction == 0:  # Up prediction
            new_position = 1
        elif prediction == 2:  # Down prediction
            new_position = -1
        else:  # Stationary prediction
            new_position = 0
        
        # Execute trade if position changes
        if new_position != position:
            # Calculate simple returns (in a real system, would use actual prices)
            # Here we use a simplified approach: 
            # - If prediction is correct, earn 0.1% per trade
            # - If prediction is wrong, lose 0.1% per trade
            # - If prediction is stationary and correct, earn 0.02%
            
            if position != 0:  # Close existing position
                correct_prediction = (position == 1 and targets[idx] == 0) or (position == -1 and targets[idx] == 2)
                pnl = 0.001 * capital if correct_prediction else -0.001 * capital
                capital += pnl
                
                trade_history.append({
                    'idx': idx,
                    'action': 'close',
                    'position': position,
                    'prediction': predictions[idx-1],
                    'actual': targets[idx-1],
                    'pnl': pnl,
                    'capital': capital
                })
            
            position = new_position
            
            if position != 0:  # Open new position
                trade_history.append({
                    'idx': idx,
                    'action': 'open',
                    'position': position,
                    'prediction': prediction,
                    'capital': capital
                })
        
        idx += 1
    
    # Close final position
    if position != 0 and idx > 0:
        correct_prediction = (position == 1 and targets[idx-1] == 0) or (position == -1 and targets[idx-1] == 2)
        pnl = 0.001 * capital if correct_prediction else -0.001 * capital
        capital += pnl
        
        trade_history.append({
            'idx': idx-1,
            'action': 'close',
            'position': position,
            'prediction': predictions[idx-1],
            'actual': targets[idx-1],
            'pnl': pnl,
            'capital': capital
        })
    
    # Calculate returns
    total_return = (capital - initial_capital) / initial_capital
    
    # Calculate daily returns (approximation)
    if len(trade_history) > 0:
        trades_per_day = len(trade_history) / 10  # Assuming 10 days of test data
        daily_return = (1 + total_return) ** (1 / 10) - 1
    else:
        daily_return = 0
    
    # Calculate annualized return
    annual_return = (1 + daily_return) ** 252 - 1
    
    # Count profitable trades
    profitable_trades = sum(1 for trade in trade_history if 'pnl' in trade and trade['pnl'] > 0)
    total_closed_trades = sum(1 for trade in trade_history if 'pnl' in trade)
    win_rate = profitable_trades / total_closed_trades if total_closed_trades > 0 else 0
    
    returns = {
        'initial_capital': initial_capital,
        'final_capital': capital,
        'total_return': total_return,
        'daily_return': daily_return,
        'annual_return': annual_return,
        'total_trades': total_closed_trades,
        'profitable_trades': profitable_trades,
        'win_rate': win_rate,
        'trade_history': trade_history
    }
    
    # Print summary
    print(f'Initial Capital: ${initial_capital:.2f}')
    print(f'Final Capital: ${capital:.2f}')
    print(f'Total Return: {total_return:.2%}')
    print(f'Annualized Return: {annual_return:.2%}')
    print(f'Total Trades: {total_closed_trades}')
    print(f'Win Rate: {win_rate:.2%}')
    
    return returns


def plot_strategy_returns(returns):
    """
    Plot strategy returns
    
    Parameters:
    -----------
    returns : dict
        Strategy returns
    """
    # Extract capital history from trade history
    capital_history = [returns['initial_capital']]
    idx_history = [0]
    
    for trade in returns['trade_history']:
        if 'pnl' in trade:
            capital_history.append(trade['capital'])
            idx_history.append(trade['idx'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(idx_history, capital_history, marker='o', markersize=3)
    plt.xlabel('Trade Index')
    plt.ylabel('Capital ($)')
    plt.title('Trading Strategy Performance')
    plt.grid(True)
    plt.savefig('strategy_returns.png')
    plt.show()


def main():
    """
    Main function to run the TLo-NBoF model on simulated LOB data
    """
    # Parameters
    num_codewords = 256
    num_temporal_regions = 3
    num_filters = 256
    kernel_size = 5
    batch_size = 128
    num_epochs = 20
    learning_rate = 1e-4
    weight_decay = 1e-4
    time_steps = 15
    prediction_horizon = 10
    price_change_threshold = 0.0001
    
    # Generate simulated LOB data
    print("Generating simulated LOB data...")
    lob_data = generate_simulated_lob_data(num_days=10, events_per_day=5000, lob_levels=10)  # Reduced for faster execution
    
    # Process data
    print("Processing LOB data...")
    processor = LOBDataProcessor(time_steps=time_steps, 
                                prediction_horizon=prediction_horizon,
                                price_change_threshold=price_change_threshold)
    
    # Use the first day for training
    train_data = lob_data[0]
    X_train, y_train = processor.process_lob_data(train_data, is_training=True)
    
    # Use the second day for validation
    val_data = lob_data[1]
    X_val, y_val = processor.process_lob_data(val_data, is_training=False)
    
    # Use days 3-10 for testing (mirroring the anchored evaluation setup in the paper)
    X_test_list = []
    y_test_list = []
    
    for day in range(2, 10):
        X_day, y_day = processor.process_lob_data(lob_data[day], is_training=False)
        X_test_list.append(X_day)
        y_test_list.append(y_day)
    
    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)
    
    print(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Create datasets and dataloaders
    train_dataset = LOBDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    
    val_dataset = LOBDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    
    test_dataset = LOBDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Create model
    print("Creating TLo-NBoF model...")
    model = TemporalLoNBoF(
        input_shape=(time_steps, X_train.shape[2]),
        num_codewords=num_codewords,
        num_temporal_regions=num_temporal_regions,
        num_filters=num_filters,
        kernel_size=kernel_size,
        use_deep_features=True,
        use_temporal_modeling=True,
        use_kernel_param_learning=True,
        use_adaptive_scaling=True
    ).to(device)
    
    print(model)
    
    # Train model
    print("Training model...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, test_loader)
    
    # Plot confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'])
    
    # Implement and evaluate trading strategy
    print("Implementing trading strategy...")
    strategy_returns = implement_trading_strategy(model, test_loader, metrics)
    
    # Plot strategy returns
    plot_strategy_returns(strategy_returns)
    
    # Save model
    torch.save(model.state_dict(), 'tlonbof_model.pth')
    print("Model saved to tlonbof_model.pth")


if __name__ == "__main__":
    main()