import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import seaborn as sns
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Function to generate more challenging and realistic financial time-series data
def generate_financial_data(n_samples=5000, seq_length=30, n_features=1):
    """
    Generate more challenging financial time-series data with three trends
    but with subtler patterns and more noise.
    """
    X = np.zeros((n_samples, seq_length, n_features))
    y = np.zeros(n_samples, dtype=int)
    delta_x = np.zeros(n_samples)
    
    # Generate different patterns
    for i in range(n_samples):
        # Randomly choose a trend: 0 (downward), 1 (still), 2 (upward)
        trend = np.random.randint(0, 3)
        y[i] = trend
        
        # Create a more challenging base sequence with strong mean reversion
        # and auto-regressive components to mimic real financial data
        base = np.zeros(seq_length)
        base[0] = np.random.randn() * 0.1
        
        # AR(1) process with mean reversion
        for t in range(1, seq_length):
            # Mean reversion component
            mean_reversion = -0.1 * base[t-1]
            # Random noise
            noise = 0.1 * np.random.randn()
            # Add both components
            base[t] = base[t-1] + mean_reversion + noise
        
        # Add trend component, but make it more subtle
        if trend == 0:  # Downward
            # Slight downward trend with noise
            trend_component = -0.005 * np.arange(seq_length) + 0.1 * np.random.randn(seq_length)
            X[i, :, 0] = base + trend_component
            # Only set the last value as delta for simulation
            delta_x[i] = -0.2 - 0.3 * np.random.rand()  # Negative change
        elif trend == 1:  # Still
            # Oscillating pattern around base
            trend_component = 0.05 * np.sin(np.linspace(0, 3*np.pi, seq_length)) + 0.1 * np.random.randn(seq_length)
            X[i, :, 0] = base + trend_component
            delta_x[i] = 0.1 * np.random.randn()  # Small random change
        else:  # Upward (trend == 2)
            # Slight upward trend with noise
            trend_component = 0.005 * np.arange(seq_length) + 0.1 * np.random.randn(seq_length)
            X[i, :, 0] = base + trend_component
            delta_x[i] = 0.2 + 0.3 * np.random.rand()  # Positive change
        
        # Add some overlapping characteristics to make classification harder
        X[i, :, 0] += 0.2 * np.random.randn(seq_length)
    
    return X, y, delta_x

# Visualize a few examples from the generated data
def visualize_examples(X, y, n_examples=5):
    plt.figure(figsize=(15, 10))
    classes = ['Downward', 'Still', 'Upward']
    
    for i in range(n_examples):
        for trend in range(3):
            # Find an example of each trend
            idx = np.where(y == trend)[0][i]
            
            plt.subplot(n_examples, 3, i*3 + trend + 1)
            plt.plot(X[idx, :, 0])
            plt.title(f'Example {i+1}: {classes[trend]}')
            if i == 0:
                plt.ylabel('Price')
            if i == n_examples - 1:
                plt.xlabel('Time')
    
    plt.tight_layout()
    plt.show()

# Create a custom dataset class for financial time-series data
class FinancialDataset(Dataset):
    def __init__(self, X, y, delta_x=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.delta_x = delta_x if delta_x is None else torch.FloatTensor(delta_x)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        if self.delta_x is None:
            return self.X[idx], self.y[idx]
        return self.X[idx], self.y[idx], self.delta_x[idx]

# MSTD-RCNN Model
class MSTDRCNN(nn.Module):
    def __init__(self, input_length=30, num_scales=3, num_filters=16, hidden_size=48, num_classes=3):
        super(MSTDRCNN, self).__init__()
        self.input_length = input_length
        self.num_scales = num_scales
        self.num_filters = num_filters
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Convolutional units for different scales
        self.conv_units = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=3, padding=0)
            for _ in range(num_scales)
        ])
        
        # Output length after convolution
        self.conv_output_length = input_length - 3 + 1  # kernel size = 3, stride = 1, padding = 0
        
        # GRU layer for feature fusion
        self.gru = nn.GRU(
            input_size=num_filters * num_scales,
            hidden_size=hidden_size,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Create multi-scale inputs through downsampling
        multi_scale_inputs = []
        for scale in range(1, self.num_scales + 1):
            # Downsample by taking every scale-th element
            downsampled = x[:, ::scale, :]
            multi_scale_inputs.append(downsampled)
        
        # Extract features using CNN for each scale
        scale_features = []
        for scale_idx, scale_input in enumerate(multi_scale_inputs):
            # Reshape for 1D convolution (batch, channels, length)
            scale_input = scale_input.permute(0, 2, 1)
            
            # Apply convolution
            conv_out = self.relu(self.conv_units[scale_idx](scale_input))
            
            # Reshape back (batch, length, channels)
            conv_out = conv_out.permute(0, 2, 1)
            
            # Zero padding to align feature maps
            if scale_idx > 0:
                # Calculate padding needed
                pad_length = self.conv_output_length - conv_out.size(1)
                if pad_length > 0:
                    zero_pad = torch.zeros(batch_size, pad_length, self.num_filters, device=conv_out.device)
                    conv_out = torch.cat([zero_pad, conv_out], dim=1)
            
            scale_features.append(conv_out)
        
        # Concatenate features from different scales
        concatenated = torch.cat([feature for feature in scale_features], dim=2)
        
        # GRU to capture temporal dependency
        gru_out, _ = self.gru(concatenated)
        
        # Use the last hidden state from GRU
        last_hidden = gru_out[:, -1, :]
        
        # Fully connected layers
        fc1_out = self.relu(self.fc1(last_hidden))
        output = self.fc2(fc1_out)
        
        return output

# Basic CNN model for comparison (single-scale)
class BasicCNN(nn.Module):
    def __init__(self, input_length=30, num_filters=16, num_classes=3):
        super(BasicCNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=3, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(num_filters * (input_length - 3 + 1), 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Reshape for 1D convolution (batch, channels, length)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Basic RNN model for comparison (temporal dependency only)
class BasicRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=48, num_classes=3):
        super(BasicRNN, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        _, h_n = self.gru(x)
        x = self.fc(h_n.squeeze(0))
        return x

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, early_stopping_patience=10):
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1s = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            # Handle both cases: with or without delta_x
            if len(batch) == 3:
                inputs, labels, _ = batch  # Ignore delta_x for training
            else:
                inputs, labels = batch
                
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Handle both cases: with or without delta_x
                if len(batch) == 3:
                    inputs, labels, _ = batch  # Ignore delta_x for validation
                else:
                    inputs, labels = batch
                    
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Store predictions and true labels
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Calculate metrics
        val_acc = accuracy_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred, average='weighted')
        val_accuracies.append(val_acc)
        val_f1s.append(val_f1)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    return model, {'train_losses': train_losses, 'val_losses': val_losses, 
                  'val_accuracies': val_accuracies, 'val_f1s': val_f1s}

# Function to test the model
def test_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Handle both cases: with or without delta_x
            if len(batch) == 3:
                inputs, labels, _ = batch  # Ignore delta_x for evaluation
            else:
                inputs, labels = batch
                
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Calculate metrics
    test_acc = accuracy_score(y_true, y_pred)
    test_f1 = f1_score(y_true, y_pred, average='weighted')
    conf_mat = confusion_matrix(y_true, y_pred)
    
    return test_acc, test_f1, conf_mat, y_true, y_pred

# Function to retrieve full dataset and predictions for simulated trading
def get_test_predictions(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    delta_x_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                inputs, labels, delta_x = batch
            else:
                print("Error: test_loader doesn't contain delta_x values needed for trading simulation")
                return None, None, None
                
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            delta_x_list.extend(delta_x.cpu().numpy())
    
    return np.array(y_true), np.array(y_pred), np.array(delta_x_list)

# Function to perform simulated trading
def simulated_trading(y_true, y_pred, delta_x):
    profit = 0
    trades = 0
    successful_trades = 0
    
    for i in range(len(y_true)):
        if y_pred[i] == y_true[i]:  # Only consider correct predictions
            if y_true[i] != 1:  # Only trade on directional predictions (not still)
                profit += delta_x[i]
                trades += 1
                successful_trades += 1
        elif y_pred[i] != 1 and y_true[i] != 1:  # Wrong directional prediction
            profit -= abs(delta_x[i])  # Lose money on wrong direction
            trades += 1
    
    win_rate = successful_trades / trades if trades > 0 else 0
    return profit, trades, win_rate

# Function to run baseline models
def run_baseline(X_train, y_train, X_test, y_test, delta_x_test):
    results = {}
    
    # SVM
    print("Training SVM...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    svm = SVC()
    svm.fit(X_train_flat, y_train)
    y_pred_svm = svm.predict(X_test_flat)
    
    svm_acc = accuracy_score(y_test, y_pred_svm)
    svm_f1 = f1_score(y_test, y_pred_svm, average='weighted')
    svm_profit, svm_trades, svm_win_rate = simulated_trading(y_test, y_pred_svm, delta_x_test)
    
    results['SVM'] = {
        'accuracy': svm_acc, 
        'f1': svm_f1, 
        'profit': svm_profit,
        'trades': svm_trades,
        'win_rate': svm_win_rate
    }
    
    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train_flat, y_train)
    y_pred_rf = rf.predict(X_test_flat)
    
    rf_acc = accuracy_score(y_test, y_pred_rf)
    rf_f1 = f1_score(y_test, y_pred_rf, average='weighted')
    rf_profit, rf_trades, rf_win_rate = simulated_trading(y_test, y_pred_rf, delta_x_test)
    
    results['RF'] = {
        'accuracy': rf_acc, 
        'f1': rf_f1, 
        'profit': rf_profit,
        'trades': rf_trades,
        'win_rate': rf_win_rate
    }
    
    return results

# Function to plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracies'], label='Accuracy')
    plt.plot(history['val_f1s'], label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Function to plot confusion matrix
def plot_confusion_matrix(conf_mat, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Down', 'Still', 'Up'], 
                yticklabels=['Down', 'Still', 'Up'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

# Main function to run the experiment
def main():
    # Generate data
    print("Generating financial time-series data...")
    X, y, delta_x = generate_financial_data(n_samples=58000, seq_length=30)
    
    # Visualize some examples
    visualize_examples(X, y, n_examples=3)
    
    # Print label distribution
    unique, counts = np.unique(y, return_counts=True)
    print("Label distribution:", dict(zip(unique, counts)))
    
    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp, delta_x_train, delta_x_temp = train_test_split(
        X, y, delta_x, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test, delta_x_val, delta_x_test = train_test_split(
        X_temp, y_temp, delta_x_temp, test_size=0.5, random_state=42)
    
    print(f"Dataset shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    
    # Create datasets and data loaders
    train_dataset = FinancialDataset(X_train, y_train)
    val_dataset = FinancialDataset(X_val, y_val)
    test_dataset = FinancialDataset(X_test, y_test, delta_x_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Train and evaluate the MSTD-RCNN model
    print("\nCreating and training the MSTD-RCNN model...")
    mstd_rcnn = MSTDRCNN(input_length=30, num_scales=3, num_filters=16, hidden_size=48, num_classes=3)
    mstd_rcnn = mstd_rcnn.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mstd_rcnn.parameters(), lr=0.0005)
    
    mstd_rcnn, history_mstd = train_model(
        mstd_rcnn, train_loader, val_loader, criterion, optimizer, 
        num_epochs=100, early_stopping_patience=10
    )
    
    # Plot training history
    plot_training_history(history_mstd)
    
    # Test the MSTD-RCNN model
    print("\nTesting the MSTD-RCNN model...")
    test_acc_mstd, test_f1_mstd, conf_mat_mstd, _, _ = test_model(mstd_rcnn, test_loader)
    print(f"MSTD-RCNN - Test Accuracy: {test_acc_mstd:.4f}, Test F1 Score: {test_f1_mstd:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(conf_mat_mstd, title='MSTD-RCNN Confusion Matrix')
    
    # Get predictions for trading simulation
    y_true_mstd, y_pred_mstd, delta_x_list = get_test_predictions(mstd_rcnn, test_loader)
    mstd_profit, mstd_trades, mstd_win_rate = simulated_trading(y_true_mstd, y_pred_mstd, delta_x_list)
    print(f"MSTD-RCNN - Profit: {mstd_profit:.2f}, Trades: {mstd_trades}, Win Rate: {mstd_win_rate:.4f}")
    
    # Train and evaluate the Basic CNN model (for comparison)
    print("\nCreating and training the Basic CNN model...")
    basic_cnn = BasicCNN(input_length=30, num_filters=16, num_classes=3)
    basic_cnn = basic_cnn.to(device)
    
    optimizer_cnn = optim.Adam(basic_cnn.parameters(), lr=0.0005)
    
    basic_cnn, history_cnn = train_model(
        basic_cnn, train_loader, val_loader, criterion, optimizer_cnn, 
        num_epochs=100, early_stopping_patience=10
    )
    
    # Test the Basic CNN model
    print("\nTesting the Basic CNN model...")
    test_acc_cnn, test_f1_cnn, conf_mat_cnn, _, _ = test_model(basic_cnn, test_loader)
    print(f"Basic CNN - Test Accuracy: {test_acc_cnn:.4f}, Test F1 Score: {test_f1_cnn:.4f}")
    
    # Get predictions for trading simulation
    y_true_cnn, y_pred_cnn, delta_x_list = get_test_predictions(basic_cnn, test_loader)
    cnn_profit, cnn_trades, cnn_win_rate = simulated_trading(y_true_cnn, y_pred_cnn, delta_x_list)
    print(f"Basic CNN - Profit: {cnn_profit:.2f}, Trades: {cnn_trades}, Win Rate: {cnn_win_rate:.4f}")
    
    # Train and evaluate the Basic RNN model (for comparison)
    print("\nCreating and training the Basic RNN model...")
    basic_rnn = BasicRNN(input_size=1, hidden_size=48, num_classes=3)
    basic_rnn = basic_rnn.to(device)
    
    optimizer_rnn = optim.Adam(basic_rnn.parameters(), lr=0.0005)
    
    basic_rnn, history_rnn = train_model(
        basic_rnn, train_loader, val_loader, criterion, optimizer_rnn, 
        num_epochs=100, early_stopping_patience=10
    )
    
    # Test the Basic RNN model
    print("\nTesting the Basic RNN model...")
    test_acc_rnn, test_f1_rnn, conf_mat_rnn, _, _ = test_model(basic_rnn, test_loader)
    print(f"Basic RNN - Test Accuracy: {test_acc_rnn:.4f}, Test F1 Score: {test_f1_rnn:.4f}")
    
    # Get predictions for trading simulation
    y_true_rnn, y_pred_rnn, delta_x_list = get_test_predictions(basic_rnn, test_loader)
    rnn_profit, rnn_trades, rnn_win_rate = simulated_trading(y_true_rnn, y_pred_rnn, delta_x_list)
    print(f"Basic RNN - Profit: {rnn_profit:.2f}, Trades: {rnn_trades}, Win Rate: {rnn_win_rate:.4f}")
    
    # Run baseline models
    print("\nRunning baseline models...")
    baseline_results = run_baseline(X_train, y_train, X_test, y_test, delta_x_test)
    
    # Print baseline results
    for model_name, metrics in baseline_results.items():
        print(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, F1 Score: {metrics['f1']:.4f}, Profit: {metrics['profit']:.2f}, Trades: {metrics['trades']}, Win Rate: {metrics['win_rate']:.4f}")
    
    # Compare all models
    print("\nModel Comparison:")
    models = {
        'MSTD-RCNN': {'accuracy': test_acc_mstd, 'f1': test_f1_mstd, 'profit': mstd_profit, 'trades': mstd_trades, 'win_rate': mstd_win_rate},
        'Basic CNN': {'accuracy': test_acc_cnn, 'f1': test_f1_cnn, 'profit': cnn_profit, 'trades': cnn_trades, 'win_rate': cnn_win_rate},
        'Basic RNN': {'accuracy': test_acc_rnn, 'f1': test_f1_rnn, 'profit': rnn_profit, 'trades': rnn_trades, 'win_rate': rnn_win_rate},
        'SVM': baseline_results['SVM'],
        'RF': baseline_results['RF']
    }
    
    # Create a table for comparison
    comparison_df = pd.DataFrame({
        'Model': list(models.keys()),
        'Accuracy': [models[m]['accuracy'] for m in models],
        'F1 Score': [models[m]['f1'] for m in models],
        'Profit': [models[m]['profit'] for m in models],
        'Trades': [models[m]['trades'] for m in models],
        'Win Rate': [models[m]['win_rate'] for m in models]
    })
    
    print(comparison_df)
    
    # Plot model comparison
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.bar(comparison_df['Model'], comparison_df['Accuracy'])
    plt.title('Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 2)
    plt.bar(comparison_df['Model'], comparison_df['F1 Score'])
    plt.title('F1 Score Comparison')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 3)
    plt.bar(comparison_df['Model'], comparison_df['Profit'])
    plt.title('Profit Comparison')
    plt.ylabel('Profit')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 4)
    plt.bar(comparison_df['Model'], comparison_df['Win Rate'])
    plt.title('Win Rate Comparison')
    plt.ylabel('Win Rate')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()