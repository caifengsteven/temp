import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, matthews_corrcoef
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class AttentionLayer(nn.Module):
    """Temporal attention layer for LSTM"""
    
    def __init__(self, hidden_dim, attention_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False)
        )
        
    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_dim)
        attention_weights = self.attention(lstm_output)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_weights.squeeze(-1), dim=1).unsqueeze(-1)  # (batch_size, seq_len, 1)
        
        # Weighted sum of LSTM outputs
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # (batch_size, hidden_dim)
        
        return context_vector, attention_weights

class ALSTM(nn.Module):
    """Attentive LSTM model for stock movement prediction"""
    
    def __init__(self, input_dim, hidden_dim, attention_dim):
        super(ALSTM, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Feature mapping layer
        self.feature_mapping = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_dim, attention_dim)
        
        # Prediction layer
        self.predictor = nn.Linear(hidden_dim * 2, 1)  # Concatenate attention output and last hidden state
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)
        
        # Apply feature mapping to each time step
        mapped_features = self.feature_mapping(x)  # (batch_size, seq_len, hidden_dim)
        
        # Pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(mapped_features)  # lstm_out: (batch_size, seq_len, hidden_dim)
        
        # Get attention-weighted context vector
        context_vector, _ = self.attention(lstm_out)  # context_vector: (batch_size, hidden_dim)
        
        # Get last hidden state
        last_hidden = h_n.squeeze(0)  # (batch_size, hidden_dim)
        
        # Concatenate context vector and last hidden state
        concat = torch.cat([context_vector, last_hidden], dim=1)  # (batch_size, 2*hidden_dim)
        
        # Prediction
        prediction = self.predictor(concat)  # (batch_size, 1)
        
        return prediction


class StandardALSTM:
    """Standard Attentive LSTM without adversarial training"""
    
    def __init__(self, input_dim, hidden_dim=32, attention_dim=16, alpha=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.alpha = alpha  # Weight for L2 regularization
        
        # Initialize model
        self.model = ALSTM(input_dim, hidden_dim, attention_dim).to(device)
    
    def hinge_loss(self, y_pred, y_true):
        """Hinge loss function for classification"""
        return torch.mean(torch.clamp(1 - y_pred * y_true, min=0))
    
    def l2_regularization(self):
        """L2 regularization on model weights"""
        l2_reg = 0.0
        for param in self.model.parameters():
            l2_reg += torch.norm(param, 2)
        return self.alpha * l2_reg
    
    def train(self, train_loader, val_loader=None, epochs=30, learning_rate=0.01):
        """Train the model"""
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience = 5
        counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_loss = 0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                # Forward pass
                pred = self.model(x_batch)
                loss = self.hinge_loss(pred, y_batch) + self.l2_regularization()
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for x_batch, y_batch in val_loader:
                        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                        pred = self.model(x_batch)
                        loss = self.hinge_loss(pred, y_batch) + self.l2_regularization()
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        self.model.load_state_dict(best_model_state)
                        break
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
        
        # Load the best model if validation was used
        if val_loader is not None and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return train_losses, val_losses
    
    def predict(self, test_loader):
        """Make predictions on test data"""
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                pred = self.model(x_batch)
                # Apply sign function to get class predictions
                pred_class = torch.sign(pred).cpu().numpy()
                predictions.extend(pred_class)
                actuals.extend(y_batch.cpu().numpy())
        
        return np.array(predictions).flatten(), np.array(actuals).flatten()


class AdvALSTM:
    """Implementation of Adversarial Attentive LSTM"""
    
    def __init__(self, input_dim, hidden_dim=32, attention_dim=16, epsilon=0.01, beta=0.5, alpha=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.epsilon = epsilon  # Scale of adversarial perturbations
        self.beta = beta  # Weight for adversarial loss
        self.alpha = alpha  # Weight for L2 regularization
        
        # Initialize model
        self.model = ALSTM(input_dim, hidden_dim, attention_dim).to(device)
        
    def hinge_loss(self, y_pred, y_true):
        """Hinge loss function for classification"""
        return torch.mean(torch.clamp(1 - y_pred * y_true, min=0))
    
    def l2_regularization(self):
        """L2 regularization on model weights"""
        l2_reg = 0.0
        for param in self.model.parameters():
            l2_reg += torch.norm(param, 2)
        return self.alpha * l2_reg
    
    def train(self, train_loader, val_loader=None, epochs=30, learning_rate=0.01):
        """Train the model with adversarial training"""
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience = 5
        counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_loss = 0
            
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                # Clean examples forward pass
                pred_clean = self.model(x_batch)
                loss_clean = self.hinge_loss(pred_clean, y_batch)
                
                # Generate adversarial examples
                # We need gradients w.r.t. input
                x_adv = x_batch.clone().detach().requires_grad_(True)
                
                # Forward pass to get gradients
                pred_adv = self.model(x_adv)
                loss_adv = self.hinge_loss(pred_adv, y_batch)
                
                # Backward pass to get gradients
                self.model.zero_grad()
                loss_adv.backward()
                
                # FGSM to generate adversarial examples
                grad = x_adv.grad.sign()
                x_adv = x_batch + self.epsilon * grad
                x_adv = x_adv.detach()  # Detach from computation graph
                
                # Forward pass with adversarial examples
                pred_adv = self.model(x_adv)
                loss_adv = self.hinge_loss(pred_adv, y_batch)
                
                # Total loss
                reg_loss = self.l2_regularization()
                total_loss = loss_clean + self.beta * loss_adv + reg_loss
                
                # Backward pass and optimization
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for x_batch, y_batch in val_loader:
                        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                        pred = self.model(x_batch)
                        loss = self.hinge_loss(pred, y_batch) + self.l2_regularization()
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        self.model.load_state_dict(best_model_state)
                        break
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
        
        # Load the best model if validation was used
        if val_loader is not None and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return train_losses, val_losses
    
    def predict(self, test_loader):
        """Make predictions on test data"""
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                pred = self.model(x_batch)
                # Apply sign function to get class predictions
                pred_class = torch.sign(pred).cpu().numpy()
                predictions.extend(pred_class)
                actuals.extend(y_batch.cpu().numpy())
        
        return np.array(predictions).flatten(), np.array(actuals).flatten()


def generate_simulated_stock_data(n_stocks=10, n_days=500, seed=42):
    """
    Generate simulated stock price data.
    
    Parameters:
    -----------
    n_stocks : int
        Number of stocks to simulate
    n_days : int
        Number of days to simulate
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing simulated stock prices
    """
    np.random.seed(seed)
    
    # Initialize price data
    prices = np.zeros((n_days, n_stocks))
    
    # Set initial prices
    prices[0] = 100 * np.ones(n_stocks) + np.random.randn(n_stocks) * 10
    
    # Generate random price movements
    for i in range(1, n_days):
        # Market factor (affects all stocks)
        market_factor = np.random.normal(0, 0.005)
        
        # Individual stock factors
        for j in range(n_stocks):
            # Stock-specific movement
            stock_factor = np.random.normal(0, 0.01)
            
            # Momentum factor (slight tendency to follow previous movement)
            momentum = 0.2 * (prices[i-1, j] - (prices[i-2, j] if i > 1 else prices[i-1, j])) / prices[i-1, j]
            
            # Price change considering market, stock-specific, and momentum factors
            price_change = market_factor + stock_factor + momentum
            
            # Add some noise
            price_change += np.random.normal(0, 0.002)
            
            # Update price
            prices[i, j] = prices[i-1, j] * (1 + price_change)
    
    # Create DataFrame
    columns = [f'Stock_{j+1}' for j in range(n_stocks)]
    df = pd.DataFrame(prices, columns=columns)
    
    # Add date index
    df['Date'] = pd.date_range(start='2020-01-01', periods=n_days)
    df.set_index('Date', inplace=True)
    
    return df

def extract_features(df, window_size=10):
    """
    Extract features from stock price data as described in the paper.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing stock prices
    window_size : int
        Number of days to consider for feature extraction
        
    Returns:
    --------
    tuple
        X (features), y (labels), dates
    """
    stocks = df.columns.tolist()
    features = []
    labels = []
    dates = []
    
    for stock in stocks:
        stock_data = df[stock].values
        for i in range(window_size, len(stock_data)-1):
            # Extract window
            window = stock_data[i-window_size:i]
            
            # Calculate features as described in the paper
            open_price = window[-1]  # Use the last price in window as 'open' for the current day
            close_price = stock_data[i]  # Current day's close price
            high_price = np.max(window)
            low_price = np.min(window)
            
            # Calculate price change percentages
            c_open = open_price / close_price - 1
            c_high = high_price / close_price - 1
            c_low = low_price / close_price - 1
            
            n_close = close_price / window[-2] - 1  # Previous day's close price
            
            # Calculate moving averages
            ma_5 = np.mean(window[-5:]) / close_price - 1 if len(window) >= 5 else 0
            ma_10 = np.mean(window) / close_price - 1
            
            # Feature vector
            feature = [c_open, c_high, c_low, n_close, ma_5, ma_10]
            
            # Label: 1 if price goes up, -1 if price goes down
            next_price = stock_data[i+1]
            threshold = 0.005  # 0.5% threshold as mentioned in the paper
            if (next_price / close_price - 1) >= threshold:
                label = 1
            elif (next_price / close_price - 1) <= -threshold:
                label = -1
            else:
                continue  # Skip examples with small price movements
            
            features.append(feature)
            labels.append(label)
            dates.append(df.index[i])
    
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    return X, y, dates

def create_sequences(X, y, time_steps):
    """
    Create sequences for LSTM input.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Features
    y : numpy.ndarray
        Labels
    time_steps : int
        Number of time steps in each sequence
        
    Returns:
    --------
    tuple
        X_seq (sequence features), y_seq (labels)
    """
    X_seq = []
    y_seq = []
    
    for i in range(len(X) - time_steps + 1):
        X_seq.append(X[i:i+time_steps])
        y_seq.append(y[i+time_steps-1])
    
    return np.array(X_seq), np.array(y_seq)

def evaluate_model(y_pred, y_true):
    """
    Evaluate model performance.
    
    Parameters:
    -----------
    y_pred : numpy.ndarray
        Predicted labels
    y_true : numpy.ndarray
        True labels
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    return {
        'Accuracy': accuracy * 100,  # Convert to percentage
        'MCC': mcc
    }

def plot_losses(train_losses, val_losses=None, title="Training Loss"):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    if val_losses is not None:
        plt.plot(val_losses, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def create_adversarial_examples(model, dataloader, epsilon=0.05):
    """
    Create adversarial examples for testing using random perturbations.
    
    Parameters:
    -----------
    model : StandardALSTM or AdvALSTM
        Model to test
    dataloader : torch.utils.data.DataLoader
        DataLoader containing test data
    epsilon : float
        Perturbation magnitude
    
    Returns:
    --------
    tuple
        Dataloader with adversarial examples, tensor of true labels
    """
    all_x_adv = []
    all_y = []
    
    # Create random perturbations
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        
        # Create random perturbation
        perturbation = torch.randn_like(x_batch).sign() * epsilon
        
        # Add perturbation to input
        x_adv = x_batch + perturbation
        
        all_x_adv.append(x_adv.cpu())
        all_y.append(y_batch)
    
    # Combine batches
    all_x_adv = torch.cat(all_x_adv, dim=0)
    all_y = torch.cat(all_y, dim=0)
    
    # Create new dataset and dataloader
    adv_dataset = TensorDataset(all_x_adv, all_y)
    adv_dataloader = DataLoader(adv_dataset, batch_size=dataloader.batch_size)
    
    return adv_dataloader, all_y

def main():
    print("Generating simulated stock data...")
    df = generate_simulated_stock_data(n_stocks=20, n_days=1000)
    
    print("Extracting features...")
    X, y, dates = extract_features(df, window_size=10)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create sequences
    time_steps = 5
    X_seq, y_seq = create_sequences(X_scaled, y, time_steps)
    
    print(f"Total sequences: {len(X_seq)}")
    print(f"Feature shape: {X_seq.shape}")
    print(f"Label shape: {y_seq.shape}")
    print(f"Positive examples: {sum(y_seq > 0)}, Negative examples: {sum(y_seq < 0)}")
    
    # Split into train, validation, and test sets
    train_ratio = 0.6
    val_ratio = 0.2
    
    train_size = int(len(X_seq) * train_ratio)
    val_size = int(len(X_seq) * val_ratio)
    
    X_train, y_train = X_seq[:train_size], y_seq[:train_size]
    X_val, y_val = X_seq[train_size:train_size+val_size], y_seq[train_size:train_size+val_size]
    X_test, y_test = X_seq[train_size+val_size:], y_seq[train_size+val_size:]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Convert to PyTorch tensors and create DataLoaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
    )
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Train standard ALSTM
    print("\nTraining standard ALSTM model...")
    alstm = StandardALSTM(
        input_dim=X_train.shape[2],
        hidden_dim=32,
        attention_dim=16,
        alpha=0.01
    )
    alstm_train_losses, alstm_val_losses = alstm.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        learning_rate=0.01
    )
    
    # Train Adversarial ALSTM
    print("\nTraining Adversarial ALSTM model...")
    adv_alstm = AdvALSTM(
        input_dim=X_train.shape[2],
        hidden_dim=32,
        attention_dim=16,
        epsilon=0.01,
        beta=0.5,
        alpha=0.01
    )
    adv_train_losses, adv_val_losses = adv_alstm.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        learning_rate=0.01
    )
    
    # Plot training and validation losses
    plot_losses(alstm_train_losses, alstm_val_losses, "Standard ALSTM: Training and Validation Loss")
    plot_losses(adv_train_losses, adv_val_losses, "Adversarial ALSTM: Training and Validation Loss")
    
    # Evaluate on test set
    print("\nEvaluating models on test set...")
    alstm_pred, alstm_true = alstm.predict(test_loader)
    adv_alstm_pred, adv_alstm_true = adv_alstm.predict(test_loader)
    
    alstm_metrics = evaluate_model(alstm_pred, alstm_true)
    adv_alstm_metrics = evaluate_model(adv_alstm_pred, adv_alstm_true)
    
    print("\nStandard ALSTM Results:")
    for metric, value in alstm_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nAdversarial ALSTM Results:")
    for metric, value in adv_alstm_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Test robustness against adversarial examples
    print("\nTesting robustness against adversarial examples...")
    
    # Create adversarial examples
    adv_test_loader, adv_true_labels = create_adversarial_examples(alstm, test_loader, epsilon=0.05)
    
    # Evaluate on adversarial examples
    alstm_adv_pred, _ = alstm.predict(adv_test_loader)
    adv_alstm_adv_pred, _ = adv_alstm.predict(adv_test_loader)
    
    alstm_adv_metrics = evaluate_model(alstm_adv_pred, adv_true_labels.numpy())
    adv_alstm_adv_metrics = evaluate_model(adv_alstm_adv_pred, adv_true_labels.numpy())
    
    print("\nStandard ALSTM Results on Adversarial Examples:")
    for metric, value in alstm_adv_metrics.items():
        original_value = alstm_metrics[metric]
        decrease = original_value - value
        print(f"{metric}: {value:.4f} (decrease of {decrease:.4f})")
    
    print("\nAdversarial ALSTM Results on Adversarial Examples:")
    for metric, value in adv_alstm_adv_metrics.items():
        original_value = adv_alstm_metrics[metric]
        decrease = original_value - value
        print(f"{metric}: {value:.4f} (decrease of {decrease:.4f})")
    
    # Calculate prediction changes due to adversarial examples
    alstm_changes = np.mean(alstm_pred != alstm_adv_pred) * 100
    adv_alstm_changes = np.mean(adv_alstm_pred != adv_alstm_adv_pred) * 100
    
    print(f"\nPrediction changes due to adversarial examples:")
    print(f"Standard ALSTM: {alstm_changes:.2f}% of predictions changed")
    print(f"Adversarial ALSTM: {adv_alstm_changes:.2f}% of predictions changed")
    
    # Visualize predictions
    plt.figure(figsize=(15, 6))
    
    # Actual vs Predicted
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(alstm_true[:100])), alstm_true[:100], color='blue', marker='o', label='Actual')
    plt.scatter(range(len(alstm_pred[:100])), alstm_pred[:100], color='green', marker='x', label='ALSTM')
    plt.scatter(range(len(adv_alstm_pred[:100])), adv_alstm_pred[:100], color='red', marker='+', label='Adv-ALSTM')
    plt.title('Actual vs Predicted (First 100 Samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Stock Movement (1/-1)')
    plt.grid(True)
    plt.legend()
    
    # Prediction Difference
    plt.subplot(1, 2, 2)
    diff = np.abs(alstm_pred - alstm_true) - np.abs(adv_alstm_pred - adv_alstm_true)
    plt.hist(diff, bins=3, alpha=0.7)
    plt.title('Prediction Error Difference (ALSTM - Adv-ALSTM)')
    plt.xlabel('Error Difference')
    plt.ylabel('Count')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Additional visualization for adversarial robustness
    plt.figure(figsize=(12, 6))
    
    # Standard ALSTM clean vs adversarial
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(alstm_pred[:50])), alstm_pred[:50], color='blue', marker='o', label='Clean')
    plt.scatter(range(len(alstm_adv_pred[:50])), alstm_adv_pred[:50], color='red', marker='x', label='Adversarial')
    plt.title('Standard ALSTM: Clean vs. Adversarial Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('Prediction (1/-1)')
    plt.grid(True)
    plt.legend()
    
    # Adversarial ALSTM clean vs adversarial
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(adv_alstm_pred[:50])), adv_alstm_pred[:50], color='blue', marker='o', label='Clean')
    plt.scatter(range(len(adv_alstm_adv_pred[:50])), adv_alstm_adv_pred[:50], color='red', marker='x', label='Adversarial')
    plt.title('Adversarial ALSTM: Clean vs. Adversarial Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('Prediction (1/-1)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()