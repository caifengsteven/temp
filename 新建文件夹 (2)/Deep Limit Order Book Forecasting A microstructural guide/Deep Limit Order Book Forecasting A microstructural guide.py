import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, LSTM, Concatenate, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import matthews_corrcoef, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Parameters for LOB simulation
num_price_levels = 10  # Number of price levels in the LOB
time_steps = 100  # Number of time steps for each sample
n_features = 4 * num_price_levels  # ask_prices, ask_volumes, bid_prices, bid_volumes
num_samples = 5000  # Number of samples to generate
prediction_horizon = 10  # Prediction horizon (H10 in the paper)
tick_size = 0.01  # Minimum price increment

# Define different stock types
stock_types = {
    'large_tick': {'initial_price': 50.0, 'avg_spread': 0.01, 'volatility': 0.0005, 'avg_volume': 1000},
    'medium_tick': {'initial_price': 100.0, 'avg_spread': 0.025, 'volatility': 0.001, 'avg_volume': 500},
    'small_tick': {'initial_price': 500.0, 'avg_spread': 0.05, 'volatility': 0.002, 'avg_volume': 200}
}

# Function to simulate a limit order book
def simulate_lob(stock_type, num_samples, time_steps, num_price_levels, tick_size):
    params = stock_types[stock_type]
    initial_price = params['initial_price']
    avg_spread = params['avg_spread']
    volatility = params['volatility']
    avg_volume = params['avg_volume']
    
    # Arrays to store LOB data and mid-price changes
    lob_data = np.zeros((num_samples, time_steps, n_features))
    mid_price_changes = np.zeros(num_samples)
    
    for i in range(num_samples):
        # Initialize mid price
        mid_price = initial_price
        
        # Simulate price movements
        price_movements = np.random.normal(0, volatility, time_steps)
        prices = np.cumsum(price_movements) + mid_price
        
        # Create LOB for each time step
        for t in range(time_steps):
            current_mid = prices[t]
            
            # Generate spread (tends to be larger for small-tick stocks)
            current_spread = max(tick_size, np.random.exponential(avg_spread))
            
            # Calculate best ask and best bid
            best_ask = current_mid + current_spread / 2
            best_bid = current_mid - current_spread / 2
            
            # Generate price levels
            ask_prices = np.array([best_ask + j * tick_size for j in range(num_price_levels)])
            bid_prices = np.array([best_bid - j * tick_size for j in range(num_price_levels)])
            
            # Generate volumes (decreasing with distance from mid price)
            ask_volumes = np.array([
                np.random.poisson(avg_volume * np.exp(-j * 0.1)) for j in range(num_price_levels)
            ])
            bid_volumes = np.array([
                np.random.poisson(avg_volume * np.exp(-j * 0.1)) for j in range(num_price_levels)
            ])
            
            # Store in LOB data
            lob_data[i, t, :num_price_levels] = ask_prices
            lob_data[i, t, num_price_levels:2*num_price_levels] = ask_volumes
            lob_data[i, t, 2*num_price_levels:3*num_price_levels] = bid_prices
            lob_data[i, t, 3*num_price_levels:] = bid_volumes
        
        # Calculate future mid price (after prediction_horizon steps)
        if i + prediction_horizon < num_samples:
            future_mid = (lob_data[i + prediction_horizon, 0, 0] + lob_data[i + prediction_horizon, 0, 2*num_price_levels]) / 2
            current_mid = (lob_data[i, -1, 0] + lob_data[i, -1, 2*num_price_levels]) / 2
            mid_price_change = future_mid - current_mid
            
            # Create labels as per the paper: -1 (down), 0 (stable), 1 (up)
            if mid_price_change <= -tick_size:
                mid_price_changes[i] = -1  # Down
            elif mid_price_change >= tick_size:
                mid_price_changes[i] = 1   # Up
            else:
                mid_price_changes[i] = 0   # Stable
        else:
            # For the last few samples, we'll just use random labels
            mid_price_changes[i] = np.random.choice([-1, 0, 1])
    
    return lob_data, mid_price_changes

# Create a simplified version of the DeepLOB model
def create_deeplob_model(input_shape, num_classes=3):
    input_layer = Input(shape=input_shape)
    
    # Convolutional layers
    conv1 = Conv1D(filters=16, kernel_size=2, strides=1, activation='relu')(input_layer)
    conv2 = Conv1D(filters=16, kernel_size=4, strides=1, activation='relu')(conv1)
    conv3 = Conv1D(filters=16, kernel_size=8, strides=1, activation='relu')(conv2)
    
    # Pooling layer
    pool = MaxPooling1D(pool_size=2)(conv3)
    
    # LSTM layer
    lstm = LSTM(64, return_sequences=False)(pool)
    
    # Output layer
    output = Dense(num_classes, activation='softmax')(lstm)
    
    model = Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Prepare the data for model training
def prepare_data(lob_data, labels, train_ratio=0.7, val_ratio=0.15):
    num_samples = lob_data.shape[0]
    
    # Normalize the data (feature-wise z-score normalization)
    reshaped_data = lob_data.reshape(-1, lob_data.shape[-1])
    scaler = StandardScaler()
    reshaped_data = scaler.fit_transform(reshaped_data)
    normalized_data = reshaped_data.reshape(lob_data.shape)
    
    # Split the data
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    
    X_train = normalized_data[:train_size]
    y_train = labels[:train_size]
    
    X_val = normalized_data[train_size:train_size + val_size]
    y_val = labels[train_size:train_size + val_size]
    
    X_test = normalized_data[train_size + val_size:]
    y_test = labels[train_size + val_size:]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# Evaluate model performance
def evaluate_model(model, X_test, y_test, threshold=0.3):
    # Get model predictions
    y_prob = model.predict(X_test)
    y_pred = np.argmax(y_prob, axis=1)
    
    # Map predictions back to -1, 0, 1 format
    y_pred_mapped = y_pred - 1  # Convert 0,1,2 to -1,0,1
    
    # Calculate metrics
    confusion = confusion_matrix(y_test, y_pred_mapped)
    mcc = matthews_corrcoef(y_test, y_pred_mapped)
    
    # Apply confidence threshold
    confidences = np.max(y_prob, axis=1)
    confident_indices = np.where(confidences >= threshold)[0]
    
    if len(confident_indices) > 0:
        y_test_confident = y_test[confident_indices]
        y_pred_confident = y_pred_mapped[confident_indices]
        confusion_confident = confusion_matrix(y_test_confident, y_pred_confident)
        mcc_confident = matthews_corrcoef(y_test_confident, y_pred_confident)
    else:
        confusion_confident = np.zeros((3, 3))
        mcc_confident = 0
    
    return {
        'confusion': confusion,
        'mcc': mcc,
        'confident_samples_ratio': len(confident_indices) / len(y_test),
        'confusion_confident': confusion_confident,
        'mcc_confident': mcc_confident,
        'y_prob': y_prob,
        'y_pred': y_pred_mapped,
        'y_test': y_test
    }

# Function to calculate the probability of correctly executing a transaction
def calculate_transaction_probability(y_test, y_pred):
    # Find potential transactions in targets
    PT = 0
    for i in range(len(y_test) - 1):
        if (y_test[i] == -1 and y_test[i+1] == 1) or (y_test[i] == 1 and y_test[i+1] == -1):
            PT += 1
    
    # Find executed transactions in predictions
    TT = 0
    for i in range(len(y_pred) - 1):
        if (y_pred[i] == -1 and y_pred[i+1] == 1) or (y_pred[i] == 1 and y_pred[i+1] == -1):
            TT += 1
    
    # Find correctly executed transactions
    CT = 0
    for i in range(len(y_test) - 1):
        if ((y_test[i] == -1 and y_test[i+1] == 1) and (y_pred[i] == -1 and y_pred[i+1] == 1)) or \
           ((y_test[i] == 1 and y_test[i+1] == -1) and (y_pred[i] == 1 and y_pred[i+1] == -1)):
            CT += 1
    
    # Calculate probability
    if PT + TT - CT > 0:
        pT = CT / (PT + TT - CT)
    else:
        pT = 0
    
    return {
        'PT': PT,
        'TT': TT,
        'CT': CT,
        'pT': pT
    }

# Function to plot confusion matrix
def plot_confusion_matrix(cm, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Down', 'Stable', 'Up'],
                yticklabels=['Down', 'Stable', 'Up'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Test model on different stock types
for stock_type in ['large_tick', 'medium_tick', 'small_tick']:
    print(f"\n=== Testing on {stock_type} stock ===")
    
    # Simulate LOB data
    print(f"Simulating {stock_type} LOB data...")
    lob_data, labels = simulate_lob(stock_type, num_samples, time_steps, num_price_levels, tick_size)
    
    # Prepare data
    print("Preparing data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data(lob_data, labels)
    
    # Count class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"Training class distribution: {dict(zip(['Down', 'Stable', 'Up'], counts))}")
    
    # Create and train model
    print("Creating and training model...")
    model = create_deeplob_model(input_shape=(time_steps, n_features))
    
    # To balance the classes, we'll use class weights
    class_weights = {}
    max_count = max(counts)
    for i, count in enumerate(counts):
        class_weights[i] = max_count / count
    
    # Train the model
    model.fit(
        X_train, y_train + 1,  # Add 1 to convert labels from [-1,0,1] to [0,1,2]
        epochs=10,
        batch_size=32,
        validation_data=(X_val, y_val + 1),
        class_weight=class_weights,
        verbose=1
    )
    
    # Evaluate without threshold
    print("\nEvaluating model without threshold...")
    eval_results = evaluate_model(model, X_test, y_test, threshold=0.3)
    print(f"MCC: {eval_results['mcc']:.4f}")
    print("Confusion Matrix:")
    plot_confusion_matrix(eval_results['confusion'])
    
    # Evaluate with threshold
    print("\nEvaluating model with threshold 0.7...")
    eval_results_thresh = evaluate_model(model, X_test, y_test, threshold=0.7)
    print(f"MCC: {eval_results_thresh['mcc_confident']:.4f}")
    print(f"Confident samples ratio: {eval_results_thresh['confident_samples_ratio']:.2%}")
    print("Confusion Matrix (confident predictions):")
    if eval_results_thresh['confident_samples_ratio'] > 0:
        plot_confusion_matrix(eval_results_thresh['confusion_confident'])
    else:
        print("No samples meet the confidence threshold.")
    
    # Calculate transaction probability
    trans_prob = calculate_transaction_probability(eval_results['y_test'], eval_results['y_pred'])
    print("\nTransaction Probability Analysis:")
    print(f"Potential Transactions (PT): {trans_prob['PT']}")
    print(f"Total Executed Transactions (TT): {trans_prob['TT']}")
    print(f"Correct Transactions (CT): {trans_prob['CT']}")
    print(f"Transaction Probability (pT): {trans_prob['pT']:.4f}")
    
    # Apply multiple thresholds and see how MCC and pT change
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    mcc_values = []
    pt_values = []
    remaining_data_ratio = []
    
    print("\nEvaluating performance across different probability thresholds:")
    for threshold in thresholds:
        eval_results_t = evaluate_model(model, X_test, y_test, threshold=threshold)
        confidences = np.max(eval_results['y_prob'], axis=1)
        confident_indices = np.where(confidences >= threshold)[0]
        
        if len(confident_indices) > 0:
            y_test_confident = y_test[confident_indices]
            y_pred_confident = eval_results['y_pred'][confident_indices]
            
            mcc_t = matthews_corrcoef(y_test_confident, y_pred_confident)
            mcc_values.append(mcc_t)
            
            # Calculate transaction probability for confident predictions
            trans_prob_t = calculate_transaction_probability(y_test_confident, y_pred_confident)
            pt_values.append(trans_prob_t['pT'])
            
            remaining_data_ratio.append(len(confident_indices) / len(y_test))
        else:
            mcc_values.append(0)
            pt_values.append(0)
            remaining_data_ratio.append(0)
    
    # Plot MCC and pT vs threshold
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, mcc_values, 'o-', label='MCC')
    plt.xlabel('Threshold')
    plt.ylabel('MCC')
    plt.title(f'MCC vs Threshold ({stock_type})')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(thresholds, pt_values, 'o-', label='pT')
    plt.xlabel('Threshold')
    plt.ylabel('Transaction Probability (pT)')
    plt.title(f'pT vs Threshold ({stock_type})')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Plot remaining data ratio
    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, remaining_data_ratio, 'o-')
    plt.xlabel('Threshold')
    plt.ylabel('Remaining Data Ratio')
    plt.title(f'Remaining Data Ratio vs Threshold ({stock_type})')
    plt.grid(True)
    plt.show()

# Add a trading simulation
def simulate_trading(y_pred, prices, initial_capital=10000):
    capital = initial_capital
    position = 0  # 0: no position, 1: long, -1: short
    position_price = 0
    transaction_cost = 0.001  # 0.1% per transaction
    
    capital_history = [capital]
    position_history = [position]
    
    for i in range(1, len(y_pred)):
        current_price = prices[i]
        
        # Trading logic based on predictions
        if y_pred[i] == 1 and position <= 0:  # Predicted up and not long
            # Close short position if exists
            if position == -1:
                capital = capital + (position_price - current_price) * abs(position) * capital_history[-1]
                capital = capital * (1 - transaction_cost)  # Apply transaction cost
            
            # Open long position
            position = 1
            position_price = current_price
            
        elif y_pred[i] == -1 and position >= 0:  # Predicted down and not short
            # Close long position if exists
            if position == 1:
                capital = capital + (current_price - position_price) * position * capital_history[-1]
                capital = capital * (1 - transaction_cost)  # Apply transaction cost
            
            # Open short position
            position = -1
            position_price = current_price
        
        # Update capital based on position
        if position == 1:
            current_capital = capital + (current_price - position_price) * position * capital_history[0]
        elif position == -1:
            current_capital = capital + (position_price - current_price) * abs(position) * capital_history[0]
        else:
            current_capital = capital
        
        capital_history.append(current_capital)
        position_history.append(position)
    
    # Calculate returns
    returns = (capital_history[-1] - initial_capital) / initial_capital
    
    return {
        'capital_history': capital_history,
        'position_history': position_history,
        'returns': returns
    }

# Generate price series for trading simulation
def generate_price_series(stock_type, num_samples):
    params = stock_types[stock_type]
    initial_price = params['initial_price']
    volatility = params['volatility']
    
    # Generate price movements
    price_movements = np.random.normal(0, volatility, num_samples)
    prices = np.cumsum(price_movements) + initial_price
    
    return prices

# Run trading simulation for each stock type
for stock_type in ['large_tick', 'medium_tick', 'small_tick']:
    print(f"\n=== Trading Simulation for {stock_type} stock ===")
    
    # Simulate LOB data and get predictions
    lob_data, labels = simulate_lob(stock_type, num_samples, time_steps, num_price_levels, tick_size)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_data(lob_data, labels)
    
    # Create and train model
    model = create_deeplob_model(input_shape=(time_steps, n_features))
    
    # Train the model (with balanced classes)
    unique, counts = np.unique(y_train, return_counts=True)
    class_weights = {}
    max_count = max(counts)
    for i, count in enumerate(counts):
        class_weights[i] = max_count / count
    
    model.fit(
        X_train, y_train + 1,  # Add 1 to convert labels from [-1,0,1] to [0,1,2]
        epochs=10,
        batch_size=32,
        validation_data=(X_val, y_val + 1),
        class_weight=class_weights,
        verbose=0
    )
    
    # Get predictions
    y_prob = model.predict(X_test)
    y_pred = np.argmax(y_prob, axis=1) - 1  # Convert back to -1, 0, 1
    
    # Generate price series
    prices = generate_price_series(stock_type, len(y_test))
    
    # Run trading simulation
    trading_results = simulate_trading(y_pred, prices)
    
    # Plot capital history
    plt.figure(figsize=(10, 6))
    plt.plot(trading_results['capital_history'])
    plt.xlabel('Time Step')
    plt.ylabel('Capital')
    plt.title(f'Trading Simulation Results for {stock_type} Stock')
    plt.grid(True)
    plt.show()
    
    # Print trading performance
    print(f"Final return: {trading_results['returns']:.2%}")
    
    # Compare with different confidence thresholds
    thresholds = [0.5, 0.7, 0.9]
    returns = []
    
    for threshold in thresholds:
        # Get confident predictions
        confidences = np.max(y_prob, axis=1)
        confident_indices = np.where(confidences >= threshold)[0]
        
        if len(confident_indices) > 0:
            # Create modified predictions (only trade on confident predictions)
            y_pred_confident = np.zeros_like(y_pred)
            y_pred_confident[confident_indices] = y_pred[confident_indices]
            
            # Run trading simulation with confident predictions
            trading_results_conf = simulate_trading(y_pred_confident, prices)
            returns.append(trading_results_conf['returns'])
            
            print(f"Threshold {threshold:.1f}: Return {trading_results_conf['returns']:.2%}, "
                  f"Trading on {len(confident_indices)/len(y_pred):.2%} of signals")
        else:
            returns.append(0)
            print(f"Threshold {threshold:.1f}: No confident predictions")


