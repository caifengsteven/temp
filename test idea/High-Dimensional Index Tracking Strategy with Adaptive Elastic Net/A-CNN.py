import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define order encoding categories as described in the paper
ORDER_TYPES = ["MARKET_BUY", "MARKET_SELL", "LIMIT_BUY", "LIMIT_SELL", "CANCEL_BUY", "CANCEL_SELL"]
# Relative price categories (distance from mid-price)
PRICE_CATEGORIES = ["0-1", "1-2", "2-3", "3-5", "5-7", "7+"]
# Time difference categories (in milliseconds)
TIME_DIFF_CATEGORIES = ["0-20", "20-500", "500+"]

# Total number of categories
NUM_CATEGORIES = len(ORDER_TYPES) * len(PRICE_CATEGORIES) * len(TIME_DIFF_CATEGORIES)

class OrderBookSimulator:
    """
    Simulates order book data with properties similar to those described in the paper
    """
    def __init__(self, initial_price=100, volatility=0.02, mean_reversion=0.05, trend=0.0):
        self.mid_price = initial_price
        self.volatility = volatility  # Increased volatility for more price movement
        self.mean_reversion = mean_reversion
        self.trend = trend  # Added trend parameter for directional bias
        self.order_book = {
            'bids': {},  # price -> quantity
            'asks': {}   # price -> quantity
        }
        self.best_bid = initial_price - 0.5
        self.best_ask = initial_price + 0.5
        self.last_order_time = datetime.now()
        
        # Initialize order book with some limit orders
        for i in range(1, 20):
            self.order_book['bids'][self.best_bid - i*0.1] = np.random.poisson(100/(i*0.8))
            self.order_book['asks'][self.best_ask + i*0.1] = np.random.poisson(100/(i*0.8))
        
        # Track price history
        self.price_history = [initial_price]
        
    def update_mid_price(self):
        """Update mid price with mean-reverting random walk and trend"""
        random_component = np.random.normal(0, self.volatility)
        mean_reversion_component = self.mean_reversion * (100 - self.mid_price)
        trend_component = self.trend
        
        self.mid_price += random_component + mean_reversion_component + trend_component
        
        # Update best bid/ask
        self.best_bid = max(self.order_book['bids'].keys()) if self.order_book['bids'] else self.mid_price - 0.5
        self.best_ask = min(self.order_book['asks'].keys()) if self.order_book['asks'] else self.mid_price + 0.5
        
        return self.mid_price

    def generate_orders(self, num_orders=100, time_window=30):
        """
        Generate a sequence of orders within a time window
        
        Parameters:
        -----------
        num_orders : int
            Number of orders to generate
        time_window : int
            Time window in seconds
            
        Returns:
        --------
        list
            List of dictionaries containing order information
        """
        orders = []
        
        # Calculate time step per order
        avg_time_per_order = time_window / num_orders
        
        order_time = datetime.now()
        prev_order_time = order_time
        
        # Introduce order imbalance based on price trend
        # If price is rising, more buy orders; if falling, more sell orders
        price_change = 0
        if len(self.price_history) >= 2:
            price_change = self.price_history[-1] - self.price_history[-2]
        
        # Adjust buy/sell probability based on recent price change
        buy_prob_adjustment = np.tanh(price_change * 20) * 0.2  # Scale to ±0.2
        
        for i in range(num_orders):
            # Decide order type with probabilities
            # ~20% market orders, ~50% limit orders, ~30% cancellations
            # But adjust based on recent price movement
            order_type_rand = np.random.random()
            
            if order_type_rand < 0.1 + buy_prob_adjustment:
                order_type = "MARKET_BUY"
            elif order_type_rand < 0.2:
                order_type = "MARKET_SELL"
            elif order_type_rand < 0.45 + buy_prob_adjustment:
                order_type = "LIMIT_BUY"
            elif order_type_rand < 0.7:
                order_type = "LIMIT_SELL"
            elif order_type_rand < 0.85:
                order_type = "CANCEL_BUY"
            else:
                order_type = "CANCEL_SELL"
            
            # Decide time difference (simulate HFT patterns)
            if np.random.random() < 0.3:  # 30% HFT orders
                time_diff_ms = np.random.randint(1, 20)  # 0-20ms
            elif np.random.random() < 0.6:  # 30% algo orders
                time_diff_ms = np.random.randint(20, 500)  # 20-500ms
            else:  # 40% regular orders
                time_diff_ms = np.random.randint(500, 3000)  # >500ms
            
            # Update order time
            order_time = prev_order_time + timedelta(milliseconds=time_diff_ms)
            
            # Decide price relative to mid-price 
            # (follows power-law distribution as mentioned in the paper)
            # Exponent parameter for power law (higher means closer to mid-price)
            alpha = 2.5
            
            # Generate price difference from mid-price with power law
            x = np.random.random()
            # Inverse CDF of power law x^(-alpha)
            price_diff_ticks = int(np.ceil(x ** (-1/(alpha-1))))
            price_diff = price_diff_ticks * 0.1  # Each tick is 0.1
            
            # For buy orders, price is below mid-price; for sell orders, price is above
            if "BUY" in order_type:
                if "CANCEL" in order_type and not self.order_book['bids']:
                    continue  # Skip if no bids to cancel
                relative_price = -price_diff
            else:  # SELL
                if "CANCEL" in order_type and not self.order_book['asks']:
                    continue  # Skip if no asks to cancel
                relative_price = price_diff
            
            # Calculate actual price
            price = self.mid_price + relative_price
            
            # Update order book
            quantity = np.random.poisson(10) + 1  # Random quantity with mean 10
            
            if order_type == "MARKET_BUY":
                # Execute against lowest ask
                if self.order_book['asks']:
                    # Market buy takes liquidity from the ask side
                    best_ask_price = min(self.order_book['asks'].keys())
                    self.order_book['asks'][best_ask_price] -= quantity
                    if self.order_book['asks'][best_ask_price] <= 0:
                        del self.order_book['asks'][best_ask_price]
                    price = best_ask_price  # Transaction happens at ask price
                    
                    # Market orders move the price
                    self.mid_price = (self.mid_price * 0.95) + (price * 0.05)  # Price impact
            
            elif order_type == "MARKET_SELL":
                # Execute against highest bid
                if self.order_book['bids']:
                    # Market sell takes liquidity from the bid side
                    best_bid_price = max(self.order_book['bids'].keys())
                    self.order_book['bids'][best_bid_price] -= quantity
                    if self.order_book['bids'][best_bid_price] <= 0:
                        del self.order_book['bids'][best_bid_price]
                    price = best_bid_price  # Transaction happens at bid price
                    
                    # Market orders move the price
                    self.mid_price = (self.mid_price * 0.95) + (price * 0.05)  # Price impact
            
            elif order_type == "LIMIT_BUY":
                # Add to order book
                if price in self.order_book['bids']:
                    self.order_book['bids'][price] += quantity
                else:
                    self.order_book['bids'][price] = quantity
            
            elif order_type == "LIMIT_SELL":
                # Add to order book
                if price in self.order_book['asks']:
                    self.order_book['asks'][price] += quantity
                else:
                    self.order_book['asks'][price] = quantity
            
            elif order_type == "CANCEL_BUY":
                # Cancel bid
                if self.order_book['bids']:
                    # Find closest existing bid to target price
                    existing_prices = list(self.order_book['bids'].keys())
                    if existing_prices:
                        closest_price = min(existing_prices, key=lambda x: abs(x - price))
                        # Cancel some or all of the orders at this price
                        cancel_qty = min(self.order_book['bids'][closest_price], quantity)
                        self.order_book['bids'][closest_price] -= cancel_qty
                        if self.order_book['bids'][closest_price] <= 0:
                            del self.order_book['bids'][closest_price]
                        price = closest_price  # Use actual canceled price
            
            elif order_type == "CANCEL_SELL":
                # Cancel ask
                if self.order_book['asks']:
                    # Find closest existing ask to target price
                    existing_prices = list(self.order_book['asks'].keys())
                    if existing_prices:
                        closest_price = min(existing_prices, key=lambda x: abs(x - price))
                        # Cancel some or all of the orders at this price
                        cancel_qty = min(self.order_book['asks'][closest_price], quantity)
                        self.order_book['asks'][closest_price] -= cancel_qty
                        if self.order_book['asks'][closest_price] <= 0:
                            del self.order_book['asks'][closest_price]
                        price = closest_price  # Use actual canceled price
            
            # Update best bid/ask
            if self.order_book['bids']:
                self.best_bid = max(self.order_book['bids'].keys())
            else:
                self.best_bid = self.mid_price - 0.5
                
            if self.order_book['asks']:
                self.best_ask = min(self.order_book['asks'].keys())
            else:
                self.best_ask = self.mid_price + 0.5
            
            # Update mid-price based on best bid/ask
            old_mid = self.mid_price
            self.mid_price = (self.best_bid + self.best_ask) / 2
            
            # Also add random walk component after certain orders
            if order_type in ["MARKET_BUY", "MARKET_SELL"]:
                self.update_mid_price()
                
            # Track price history
            self.price_history.append(self.mid_price)
            
            # Calculate relative price category
            rel_price_diff = abs(price - self.mid_price)
            if rel_price_diff < 1:
                price_category = "0-1"
            elif rel_price_diff < 2:
                price_category = "1-2"
            elif rel_price_diff < 3:
                price_category = "2-3"
            elif rel_price_diff < 5:
                price_category = "3-5"
            elif rel_price_diff < 7:
                price_category = "5-7"
            else:
                price_category = "7+"
            
            # Calculate time diff category
            if time_diff_ms < 20:
                time_diff_category = "0-20"
            elif time_diff_ms < 500:
                time_diff_category = "20-500"
            else:
                time_diff_category = "500+"
            
            # Create order record
            order = {
                'timestamp': order_time,
                'type': order_type,
                'price': price,
                'relative_price': price - self.mid_price,
                'price_category': price_category,
                'quantity': quantity,
                'time_diff_ms': time_diff_ms,
                'time_diff_category': time_diff_category,
                'mid_price': self.mid_price,
                'best_bid': self.best_bid,
                'best_ask': self.best_ask
            }
            
            orders.append(order)
            prev_order_time = order_time
        
        return orders

    def encode_orders(self, orders):
        """
        Encodes orders as described in the paper
        
        Parameters:
        -----------
        orders : list
            List of order dictionaries
            
        Returns:
        --------
        numpy.ndarray
            Encoded orders as one-hot vectors
        """
        encoded_orders = np.zeros((len(orders), NUM_CATEGORIES))
        
        for i, order in enumerate(orders):
            # Find index for order type
            type_idx = ORDER_TYPES.index(order['type'])
            
            # Find index for price category
            price_idx = PRICE_CATEGORIES.index(order['price_category'])
            
            # Find index for time diff category
            time_idx = TIME_DIFF_CATEGORIES.index(order['time_diff_category'])
            
            # Calculate combined index
            combined_idx = (type_idx * len(PRICE_CATEGORIES) * len(TIME_DIFF_CATEGORIES) + 
                           price_idx * len(TIME_DIFF_CATEGORIES) + 
                           time_idx)
            
            # Set one-hot encoding
            encoded_orders[i, combined_idx] = 1
        
        return encoded_orders


class ACNNPlusModel(nn.Module):
    """
    Average Convolutional Neural Network Plus (A-CNN+) model as described in the paper
    """
    def __init__(self, input_dim, embedding_dim=5, num_pools=3, num_filters=32):
        super(ACNNPlusModel, self).__init__()
        
        self.embedding = nn.Linear(input_dim, embedding_dim)
        
        # Define different pooling sizes and convolution kernel sizes
        self.pool_sizes = [5, 10, 15][:num_pools]
        self.conv_sizes = [3, 5, 7][:num_pools]
        
        # Create convolution layers with different kernel sizes
        self.convs = nn.ModuleList()
        
        for conv_size in self.conv_sizes:
            self.convs.append(nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=conv_size,
                padding=conv_size//2  # Same padding
            ))
        
        # Final fully connected layers
        # Calculate output size of the convolutional layers
        conv_output_size = num_filters * num_pools
        
        self.fc1 = nn.Linear(conv_output_size, 64)  # Increased hidden size
        self.fc2 = nn.Linear(64, 2)  # 2 classes: up/down
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_dim)
        batch_size, seq_len, _ = x.size()
        
        # Apply embedding to each order
        # Output shape: (batch_size, sequence_length, embedding_dim)
        embedded = F.relu(self.embedding(x))
        
        # Transpose to (batch_size, embedding_dim, sequence_length) for 1D convolution
        embedded = embedded.transpose(1, 2)
        
        # Apply different pooling sizes and convolutions
        conv_outputs = []
        
        for i, (pool_size, conv) in enumerate(zip(self.pool_sizes, self.convs)):
            # Apply average pooling
            # Use F.avg_pool1d with stride=1 and appropriate padding
            padded_len = seq_len + (pool_size - 1)
            padding = (pool_size - 1) // 2
            
            # Apply average pooling
            if pool_size > 1:
                pooled = F.avg_pool1d(
                    embedded,
                    kernel_size=pool_size,
                    stride=1,
                    padding=padding
                )
            else:
                pooled = embedded
            
            # Apply convolution
            # Output shape: (batch_size, num_filters, sequence_length)
            conv_out = F.relu(conv(pooled))
            
            # Apply max pooling over the convolution output
            # Output shape: (batch_size, num_filters, 1)
            pooled_out = F.adaptive_max_pool1d(conv_out, 1)
            
            # Flatten: (batch_size, num_filters)
            pooled_out = pooled_out.squeeze(-1)
            
            conv_outputs.append(pooled_out)
        
        # Concatenate all convolution outputs
        # Output shape: (batch_size, num_filters * num_pools)
        x = torch.cat(conv_outputs, dim=1)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Apply softmax to get probabilities
        return F.softmax(x, dim=1)


class StockPricePredictionStrategy:
    """
    Strategy for predicting stock price movements using A-CNN+ model
    """
    def __init__(self, threshold=0.55, lookback_seconds=30, forecast_seconds=30, device='cpu'):
        self.model = None
        self.threshold = threshold
        self.lookback_seconds = lookback_seconds  # Time window for input data
        self.forecast_seconds = forecast_seconds  # Prediction horizon
        self.simulator = None  # Initialize in training/simulation
        self.position = 0
        self.cash = 100000
        self.trades = []
        self.portfolio_values = []
        self.predictions = []
        self.price_history = []
        self.order_history = []
        self.timestamp_history = []
        self.device = device
        
    def train_model(self, num_samples=3000, train_ratio=0.7, val_ratio=0.15, sequence_length=100,
                   batch_size=32, num_epochs=30, learning_rate=0.001, weight_decay=1e-5):
        """
        Train the A-CNN+ model with simulated data
        
        Parameters:
        -----------
        num_samples : int
            Number of samples to generate for training
        train_ratio : float
            Ratio of data to use for training
        val_ratio : float
            Ratio of data to use for validation
        sequence_length : int
            Length of order sequence for each sample
        batch_size : int
            Batch size for training
        num_epochs : int
            Number of epochs to train
        learning_rate : float
            Learning rate for optimizer
        """
        print("Generating training data...")
        X = []
        y = []
        
        # Create multiple simulators with different trends to get more variety in data
        simulators = [
            OrderBookSimulator(volatility=0.02, trend=0.01),   # Upward trend
            OrderBookSimulator(volatility=0.02, trend=-0.01),  # Downward trend
            OrderBookSimulator(volatility=0.02, trend=0.0)     # No trend
        ]
        
        for i in range(num_samples):
            if i % 100 == 0:
                print(f"Generating sample {i}/{num_samples}")
            
            # Choose a simulator randomly
            simulator = np.random.choice(simulators)
            
            # Generate order sequence
            orders = simulator.generate_orders(num_orders=sequence_length)
            
            # Encode orders
            encoded_orders = simulator.encode_orders(orders)
            
            # Record mid-price before and after
            mid_price_before = orders[-1]['mid_price']
            
            # Generate future orders to see price movement
            future_orders = simulator.generate_orders(num_orders=50)
            mid_price_after = future_orders[-1]['mid_price']
            
            # Calculate percentage change
            pct_change = (mid_price_after - mid_price_before) / mid_price_before * 100
            
            # Determine price movement (up=1, down=0)
            # Use a small threshold to filter out very small movements
            if pct_change > 0.05:  # More than 0.05% increase
                label = 1  # Up
            elif pct_change < -0.05:  # More than 0.05% decrease
                label = 0  # Down
            else:
                # Skip samples with very small changes
                continue
            
            X.append(encoded_orders)
            y.append(label)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Print class balance
        unique, counts = np.unique(y, return_counts=True)
        print(f"Class distribution: {dict(zip(unique, counts))}")
        
        # Split data
        train_size = int(len(X) * train_ratio)
        val_size = int(len(X) * val_ratio)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Build model
        print("Building model...")
        self.model = ACNNPlusModel(
            input_dim=NUM_CATEGORIES,
            embedding_dim=5,
            num_pools=3,
            num_filters=32  # Increased filters
        ).to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        
        # Training loop
        print("Training model...")
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        best_val_loss = float('inf')
        best_model_state = None
        patience = 8
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_loss /= len(train_loader.dataset)
            train_accuracy = train_correct / train_total
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss /= len(val_loader.dataset)
            val_accuracy = val_correct / val_total
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            # Update learning rate scheduler
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Evaluate on test set
        print("Evaluating model...")
        self.model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(outputs.cpu().numpy())
        
        test_loss /= len(test_loader.dataset)
        test_accuracy = test_correct / test_total
        
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        # Additional metrics
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=['Down', 'Up']))
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Training Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Analyze threshold impact on test set
        self._analyze_confidence_thresholds(X_test_tensor, y_test_tensor)
        
        # Initialize simulator for future use
        self.simulator = OrderBookSimulator(volatility=0.02)
        
        # Return test data for further analysis
        return X_test, y_test, all_probs
    
    def _analyze_confidence_thresholds(self, X_test, y_test):
        """
        Analyze the impact of confidence thresholds on prediction accuracy
        
        Parameters:
        -----------
        X_test : torch.Tensor
            Test input data
        y_test : torch.Tensor
            Test labels
        """
        print("\nAnalyzing impact of confidence thresholds...")
        
        # Generate predictions with confidence levels
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test)
        
        # Convert to numpy for analysis
        probs = predictions.cpu().numpy()
        true_labels = y_test.cpu().numpy()
        
        # Get max probability and predicted class
        confidences = np.max(probs, axis=1)
        pred_classes = np.argmax(probs, axis=1)
        
        # Analyze different confidence thresholds
        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
        results = []
        
        for threshold in thresholds:
            # Filter by confidence
            mask = confidences >= threshold
            
            if np.sum(mask) > 0:  # Only if we have predictions
                # Calculate accuracy, precision, etc.
                filtered_preds = pred_classes[mask]
                filtered_labels = true_labels[mask]
                
                accuracy = np.mean(filtered_preds == filtered_labels)
                precision = precision_score(filtered_labels, filtered_preds, average='weighted', zero_division=0)
                recall = recall_score(filtered_labels, filtered_preds, average='weighted', zero_division=0)
                f1 = f1_score(filtered_labels, filtered_preds, average='weighted', zero_division=0)
                
                results.append({
                    'threshold': threshold,
                    'coverage': np.mean(mask) * 100,  # Percentage of predictions kept
                    'predictions': np.sum(mask),  # Number of predictions
                    'accuracy': accuracy * 100,
                    'precision': precision * 100,
                    'recall': recall * 100,
                    'f1': f1 * 100
                })
            else:
                results.append({
                    'threshold': threshold,
                    'coverage': 0,
                    'predictions': 0,
                    'accuracy': 0,
                    'precision': 0,
                    'recall': 0,
                    'f1': 0
                })
        
        # Create DataFrame and plot
        df = pd.DataFrame(results)
        print("\nThreshold Analysis Results:")
        print(df.to_string(index=False))
        
        # Plot results
        plt.figure(figsize=(15, 8))
        
        # Plot 1: Accuracy vs Threshold
        plt.subplot(2, 2, 1)
        plt.plot(df['threshold'], df['accuracy'], 'o-', label='Accuracy')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy vs Confidence Threshold')
        plt.grid(True)
        
        # Plot 2: Coverage vs Threshold
        plt.subplot(2, 2, 2)
        plt.plot(df['threshold'], df['coverage'], 'o-', label='Coverage')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Coverage (%)')
        plt.title('Coverage vs Confidence Threshold')
        plt.grid(True)
        
        # Plot 3: Precision/Recall vs Threshold
        plt.subplot(2, 2, 3)
        plt.plot(df['threshold'], df['precision'], 'o-', label='Precision')
        plt.plot(df['threshold'], df['recall'], 'o-', label='Recall')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Percentage (%)')
        plt.title('Precision/Recall vs Confidence Threshold')
        plt.legend()
        plt.grid(True)
        
        # Plot 4: F1 vs Threshold
        plt.subplot(2, 2, 4)
        plt.plot(df['threshold'], df['f1'], 'o-', label='F1 Score')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('F1 Score (%)')
        plt.title('F1 Score vs Confidence Threshold')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Histogram of confidence
        plt.figure(figsize=(10, 6))
        
        # Separate by correctness
        correct = confidences[pred_classes == true_labels]
        incorrect = confidences[pred_classes != true_labels]
        
        plt.hist(correct, bins=20, alpha=0.6, label='Correct Predictions', density=True)
        plt.hist(incorrect, bins=20, alpha=0.6, label='Incorrect Predictions', density=True)
        
        plt.xlabel('Confidence')
        plt.ylabel('Density')
        plt.title('Distribution of Confidence Scores')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Suggest optimal threshold based on F1 score
        best_idx = df['f1'].idxmax()
        best_threshold = df.iloc[best_idx]['threshold']
        print(f"\nRecommended confidence threshold based on F1 score: {best_threshold}")
        
        # Update strategy threshold
        self.threshold = best_threshold
    
    def predict_price_movement(self, orders):
        """
        Predict price movement based on order sequence
        
        Parameters:
        -----------
        orders : list
            List of order dictionaries
            
        Returns:
        --------
        tuple
            (Prediction (0=down, 1=up), confidence)
        """
        if self.model is None:
            print("Model not trained. Please train the model first.")
            return None, 0
        
        # Encode orders
        encoded_orders = self.simulator.encode_orders(orders)
        
        # Convert to PyTorch tensor
        inputs = torch.FloatTensor(encoded_orders).unsqueeze(0).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(inputs)
            prediction = outputs[0]
            
        # Get class and confidence
        predicted_class = torch.argmax(prediction).item()
        confidence = prediction[predicted_class].item()
        
        return predicted_class, confidence
    
    def execute_trade(self, prediction, confidence, current_price, timestamp):
        """
        Execute a trade based on prediction and confidence
        
        Parameters:
        -----------
        prediction : int
            Predicted price movement (0=down, 1=up)
        confidence : float
            Confidence in prediction
        current_price : float
            Current price
        timestamp : datetime
            Current timestamp
        """
        # Only trade if confidence exceeds threshold
        if confidence < self.threshold:
            return False
        
        # Decide action based on prediction and current position
        if prediction == 1:  # Up
            if self.position <= 0:  # No position or short
                # Buy/Cover
                quantity = 100  # Standard lot size
                cost = quantity * current_price
                
                if self.position < 0:
                    action = "COVER"
                else:
                    action = "BUY"
                
                # Execute trade
                self.cash -= cost
                self.position += quantity
                
                self.trades.append({
                    'timestamp': timestamp,
                    'action': action,
                    'price': current_price,
                    'quantity': quantity,
                    'confidence': confidence,
                    'cost': cost
                })
                
                print(f"{timestamp}: {action} {quantity} @ ${current_price:.2f} (Confidence: {confidence:.4f})")
                return True
        
        elif prediction == 0:  # Down
            if self.position >= 0:  # No position or long
                # Sell/Short
                quantity = 100  # Standard lot size
                revenue = quantity * current_price
                
                if self.position > 0:
                    action = "SELL"
                else:
                    action = "SHORT"
                
                # Execute trade
                self.cash += revenue
                self.position -= quantity
                
                self.trades.append({
                    'timestamp': timestamp,
                    'action': action,
                    'price': current_price,
                    'quantity': quantity,
                    'confidence': confidence,
                    'revenue': revenue
                })
                
                print(f"{timestamp}: {action} {quantity} @ ${current_price:.2f} (Confidence: {confidence:.4f})")
                return True
        
        return False
    
    def update_portfolio_value(self, current_price, timestamp):
        """
        Update portfolio value
        
        Parameters:
        -----------
        current_price : float
            Current price
        timestamp : datetime
            Current timestamp
        """
        # Calculate portfolio value
        position_value = self.position * current_price
        portfolio_value = self.cash + position_value
        
        self.portfolio_values.append({
            'timestamp': timestamp,
            'cash': self.cash,
            'position': self.position,
            'position_value': position_value,
            'portfolio_value': portfolio_value,
            'price': current_price
        })
    
    def run_simulation(self, duration_minutes=60, sequence_length=100, trend=None, volatility=None):
        """
        Run trading simulation
        
        Parameters:
        -----------
        duration_minutes : int
            Duration of simulation in minutes
        sequence_length : int
            Length of order sequence for each prediction
        trend : float or None
            Optional trend parameter for simulator 
        volatility : float or None
            Optional volatility parameter for simulator
        """
        if self.model is None:
            print("Model not trained. Please train the model first.")
            return
        
        # Reset simulation state
        if trend is not None or volatility is not None:
            trend_val = trend if trend is not None else 0.0
            vol_val = volatility if volatility is not None else 0.02
            print(f"Starting simulation with trend={trend_val}, volatility={vol_val}...")
            self.simulator = OrderBookSimulator(volatility=vol_val, trend=trend_val)
        else:
            print(f"Starting simulation for {duration_minutes} minutes...")
            if self.simulator is None:
                self.simulator = OrderBookSimulator(volatility=0.02)
        
        self.position = 0
        self.cash = 100000
        self.trades = []
        self.portfolio_values = []
        self.predictions = []
        self.price_history = []
        self.order_history = []
        self.timestamp_history = []
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        current_time = start_time
        step = 0
        
        # Track total predictions and trades executed
        total_predictions = 0
        trades_executed = 0
        
        while current_time < end_time:
            step += 1
            
            # Generate orders for lookback period
            orders = self.simulator.generate_orders(
                num_orders=sequence_length,
                time_window=self.lookback_seconds
            )
            
            # Record current price and time
            current_price = orders[-1]['mid_price']
            current_time = orders[-1]['timestamp']
            
            # Predict price movement
            prediction, confidence = self.predict_price_movement(orders)
            total_predictions += 1
            
            # Execute trade based on prediction
            trade_executed = self.execute_trade(prediction, confidence, current_price, current_time)
            if trade_executed:
                trades_executed += 1
            
            # Update portfolio value
            self.update_portfolio_value(current_price, current_time)
            
            # Record prediction, price and orders
            self.predictions.append({
                'timestamp': current_time,
                'prediction': prediction,
                'confidence': confidence,
                'price': current_price
            })
            
            self.price_history.append(current_price)
            self.timestamp_history.append(current_time)
            
            # Store full order history for the first 5 steps for inspection
            if step <= 5:
                self.order_history.append(orders)
            
            # Every 10 steps, print progress
            if step % 10 == 0:
                progress = (current_time - start_time).total_seconds() / (end_time - start_time).total_seconds() * 100
                print(f"Progress: {progress:.1f}% - Time: {current_time} - Price: ${current_price:.2f}")
        
        # Print final results
        final_portfolio = self.portfolio_values[-1]['portfolio_value']
        initial_portfolio = 100000
        total_return = (final_portfolio - initial_portfolio) / initial_portfolio * 100
        
        print(f"\nSimulation completed.")
        print(f"Initial portfolio value: ${initial_portfolio:.2f}")
        print(f"Final portfolio value: ${final_portfolio:.2f}")
        print(f"Total return: {total_return:.2f}%")
        print(f"Number of predictions: {total_predictions}")
        print(f"Number of trades executed: {trades_executed}")
        print(f"Execution rate: {trades_executed/total_predictions*100:.2f}%")
        
        # Plot results
        self.plot_simulation_results()
        
        # Return summary statistics
        return {
            'initial_value': initial_portfolio,
            'final_value': final_portfolio,
            'total_return': total_return,
            'num_trades': len(self.trades),
            'price_change': (self.price_history[-1] - self.price_history[0]) / self.price_history[0] * 100
        }
        
    def plot_simulation_results(self):
        """Plot simulation results"""
        # Convert timestamps to numbers for plotting
        timestamps = [t['timestamp'] for t in self.portfolio_values]
        portfolio_values = [t['portfolio_value'] for t in self.portfolio_values]
        price_history = [t['price'] for t in self.portfolio_values]
        
        # Price history
        price_timestamps = self.timestamp_history
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        # Plot 1: Portfolio Value
        ax1.plot(timestamps, portfolio_values, 'b-')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Portfolio Value Over Time')
        ax1.grid(True)
        
        # Plot 2: Price
        ax2.plot(timestamps, price_history, 'g-')
        ax2.set_ylabel('Price ($)')
        ax2.set_title('Stock Price Over Time')
        ax2.grid(True)
        
        # Plot 3: Position
        positions = [t['position'] for t in self.portfolio_values]
        ax3.plot(timestamps, positions, 'r-')
        ax3.set_ylabel('Position (shares)')
        ax3.set_xlabel('Time')
        ax3.set_title('Position Over Time')
        ax3.grid(True)
        
        # Add trade markers
        buy_times = [t['timestamp'] for t in self.trades if t['action'] in ('BUY', 'COVER')]
        buy_prices = [t['price'] for t in self.trades if t['action'] in ('BUY', 'COVER')]
        
        sell_times = [t['timestamp'] for t in self.trades if t['action'] in ('SELL', 'SHORT')]
        sell_prices = [t['price'] for t in self.trades if t['action'] in ('SELL', 'SHORT')]
        
        ax2.scatter(buy_times, buy_prices, color='green', marker='^', s=100, label='Buy/Cover')
        ax2.scatter(sell_times, sell_prices, color='red', marker='v', s=100, label='Sell/Short')
        ax2.legend()
        
        # Format x-axis
        plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        plt.show()
        
        # Additional analysis
        # Plot prediction confidence
        plt.figure(figsize=(15, 6))
        
        # Get prediction data
        pred_timestamps = [p['timestamp'] for p in self.predictions]
        pred_confidence = [p['confidence'] for p in self.predictions]
        pred_direction = [p['prediction'] for p in self.predictions]
        
        # Separate up and down predictions
        up_timestamps = [t for t, p in zip(pred_timestamps, pred_direction) if p == 1]
        up_confidence = [c for c, p in zip(pred_confidence, pred_direction) if p == 1]
        
        down_timestamps = [t for t, p in zip(pred_timestamps, pred_direction) if p == 0]
        down_confidence = [c for c, p in zip(pred_confidence, pred_direction) if p == 0]
        
        plt.scatter(up_timestamps, up_confidence, color='green', marker='o', alpha=0.7, label='Up Prediction')
        plt.scatter(down_timestamps, down_confidence, color='red', marker='o', alpha=0.7, label='Down Prediction')
        
        plt.axhline(y=self.threshold, color='black', linestyle='--', label=f'Threshold ({self.threshold})')
        
        plt.ylabel('Prediction Confidence')
        plt.title('Prediction Confidence Over Time')
        plt.legend()
        plt.grid(True)
        
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.show()
        
        # Calculate trade statistics
        if len(self.trades) > 1:
            # Profitable trades
            profits = []
            profitable_trades = 0
            
            # Create pairs of trades
            paired_trades = []
            
            i = 0
            while i < len(self.trades):
                entry = self.trades[i]
                
                # Find the matching exit trade
                j = i + 1
                while j < len(self.trades):
                    exit = self.trades[j]
                    # Check if this is a matching exit
                    if (entry['action'] in ('BUY', 'COVER') and exit['action'] in ('SELL', 'SHORT')) or \
                       (entry['action'] in ('SELL', 'SHORT') and exit['action'] in ('BUY', 'COVER')):
                        paired_trades.append((entry, exit))
                        i = j + 1  # Move past this pair
                        break
                    j += 1
                    
                # If no exit found, move to next trade
                if j >= len(self.trades):
                    i += 1
            
            # Calculate profits for paired trades
            for entry, exit in paired_trades:
                if entry['action'] in ('BUY', 'COVER') and exit['action'] in ('SELL', 'SHORT'):
                    # Long trade
                    profit = exit.get('revenue', 0) - entry.get('cost', 0)
                    profit_pct = (exit['price'] - entry['price']) / entry['price'] * 100
                elif entry['action'] in ('SELL', 'SHORT') and exit['action'] in ('BUY', 'COVER'):
                    # Short trade
                    profit = entry.get('revenue', 0) - exit.get('cost', 0)
                    profit_pct = (entry['price'] - exit['price']) / entry['price'] * 100
                else:
                    continue
                
                profits.append({'profit': profit, 'percent': profit_pct})
                if profit > 0:
                    profitable_trades += 1
            
            if profits:
                win_rate = profitable_trades / len(profits) * 100
                avg_profit = sum(p['profit'] for p in profits) / len(profits)
                avg_profit_pct = sum(p['percent'] for p in profits) / len(profits)
                max_profit = max(p['profit'] for p in profits)
                max_loss = min(p['profit'] for p in profits)
                
                print("\nTrade Statistics:")
                print(f"Number of round-trip trades: {len(profits)}")
                print(f"Win rate: {win_rate:.2f}%")
                print(f"Average profit: ${avg_profit:.2f} ({avg_profit_pct:.2f}%)")
                print(f"Maximum profit: ${max_profit:.2f}")
                print(f"Maximum loss: ${max_loss:.2f}")
                
                # Plot profit distribution
                plt.figure(figsize=(12, 6))
                plt.hist([p['profit'] for p in profits], bins=20, color='blue', alpha=0.7)
                plt.axvline(x=0, color='red', linestyle='--')
                plt.xlabel('Profit/Loss ($)')
                plt.ylabel('Frequency')
                plt.title('Distribution of Trade Profits/Losses')
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                
                # Plot trade returns vs confidence
                plt.figure(figsize=(12, 6))
                
                confidences = []
                returns = []
                
                for (entry, exit), profit_data in zip(paired_trades, profits):
                    confidences.append(entry['confidence'])
                    returns.append(profit_data['percent'])
                
                plt.scatter(confidences, returns, alpha=0.7)
                plt.axhline(y=0, color='red', linestyle='--')
                plt.xlabel('Entry Signal Confidence')
                plt.ylabel('Trade Return (%)')
                plt.title('Trade Returns vs Signal Confidence')
                plt.grid(True)
                plt.tight_layout()
                plt.show()

    def analyze_threshold_impact(self, confidence_thresholds=None):
        """
        Analyze the impact of different confidence thresholds on profitability
        
        Parameters:
        -----------
        confidence_thresholds : list
            List of confidence thresholds to test
        """
        if not self.predictions:
            print("No predictions available. Please run simulation first.")
            return
        
        if confidence_thresholds is None:
            confidence_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
        
        results = []
        
        for threshold in confidence_thresholds:
            # Reset simulation state
            position = 0
            cash = 100000
            trades = []
            portfolio_values = []
            
            for i, pred in enumerate(self.predictions):
                prediction = pred['prediction']
                confidence = pred['confidence']
                current_price = pred['price']
                timestamp = pred['timestamp']
                
                # Only trade if confidence exceeds threshold
                if confidence >= threshold:
                    # Decide action based on prediction and current position
                    if prediction == 1:  # Up
                        if position <= 0:  # No position or short
                            # Buy/Cover
                            quantity = 100  # Standard lot size
                            cost = quantity * current_price
                            
                            if position < 0:
                                action = "COVER"
                            else:
                                action = "BUY"
                            
                            # Execute trade
                            cash -= cost
                            position += quantity
                            
                            trades.append({
                                'timestamp': timestamp,
                                'action': action,
                                'price': current_price,
                                'quantity': quantity,
                                'confidence': confidence,
                                'cost': cost
                            })
                    
                    elif prediction == 0:  # Down
                        if position >= 0:  # No position or long
                            # Sell/Short
                            quantity = 100  # Standard lot size
                            revenue = quantity * current_price
                            
                            if position > 0:
                                action = "SELL"
                            else:
                                action = "SHORT"
                            
                            # Execute trade
                            cash += revenue
                            position -= quantity
                            
                            trades.append({
                                'timestamp': timestamp,
                                'action': action,
                                'price': current_price,
                                'quantity': quantity,
                                'confidence': confidence,
                                'revenue': revenue
                            })
                
                # Calculate portfolio value
                position_value = position * current_price
                portfolio_value = cash + position_value
                
                portfolio_values.append({
                    'timestamp': timestamp,
                    'cash': cash,
                    'position': position,
                    'position_value': position_value,
                    'portfolio_value': portfolio_value
                })
            
            # Close final position if necessary
            if position != 0 and len(self.predictions) > 0:
                final_price = self.predictions[-1]['price']
                
                if position > 0:
                    # Sell
                    revenue = position * final_price
                    cash += revenue
                    
                    trades.append({
                        'timestamp': self.predictions[-1]['timestamp'],
                        'action': "SELL",
                        'price': final_price,
                        'quantity': position,
                        'confidence': 1.0,
                        'revenue': revenue
                    })
                else:
                    # Cover
                    cost = abs(position) * final_price
                    cash -= cost
                    
                    trades.append({
                        'timestamp': self.predictions[-1]['timestamp'],
                        'action': "COVER",
                        'price': final_price,
                        'quantity': abs(position),
                        'confidence': 1.0,
                        'cost': cost
                    })
                
                # Reset position
                position = 0
                
                # Update final portfolio value
                portfolio_values.append({
                    'timestamp': self.predictions[-1]['timestamp'],
                    'cash': cash,
                    'position': position,
                    'position_value': 0,
                    'portfolio_value': cash
                })
            
            # Calculate statistics
            initial_portfolio = 100000
            final_portfolio = portfolio_values[-1]['portfolio_value'] if portfolio_values else initial_portfolio
            total_return = (final_portfolio - initial_portfolio) / initial_portfolio * 100
            num_trades = len(trades)
            
            # Calculate win rate for trades
            win_rate = 0
            avg_profit = 0
            
            if num_trades > 1:
                # Create pairs of trades
                paired_trades = []
                
                i = 0
                while i < len(trades):
                    entry = trades[i]
                    
                    # Find the matching exit trade
                    j = i + 1
                    while j < len(trades):
                        exit = trades[j]
                        # Check if this is a matching exit
                        if (entry['action'] in ('BUY', 'COVER') and exit['action'] in ('SELL', 'SHORT')) or \
                           (entry['action'] in ('SELL', 'SHORT') and exit['action'] in ('BUY', 'COVER')):
                            paired_trades.append((entry, exit))
                            i = j + 1  # Move past this pair
                            break
                        j += 1
                        
                    # If no exit found, move to next trade
                    if j >= len(trades):
                        i += 1
                
                # Calculate profits for paired trades
                profits = []
                profitable_trades = 0
                
                for entry, exit in paired_trades:
                    if entry['action'] in ('BUY', 'COVER') and exit['action'] in ('SELL', 'SHORT'):
                        # Long trade
                        profit = exit.get('revenue', 0) - entry.get('cost', 0)
                    elif entry['action'] in ('SELL', 'SHORT') and exit['action'] in ('BUY', 'COVER'):
                        # Short trade
                        profit = entry.get('revenue', 0) - exit.get('cost', 0)
                    else:
                        continue
                    
                    profits.append(profit)
                    if profit > 0:
                        profitable_trades += 1
                
                if profits:
                    win_rate = profitable_trades / len(profits) * 100
                    avg_profit = sum(profits) / len(profits)
            
            results.append({
                'threshold': threshold,
                'final_portfolio': final_portfolio,
                'total_return': total_return,
                'num_trades': num_trades,
                'num_round_trips': len(paired_trades) if 'paired_trades' in locals() else 0,
                'win_rate': win_rate,
                'avg_profit': avg_profit
            })
        
        # Create DataFrame and plot results
        df = pd.DataFrame(results)
        
        # Create plots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Total Return vs Threshold
        ax1.plot(df['threshold'], df['total_return'], 'bo-')
        ax1.set_ylabel('Total Return (%)')
        ax1.set_title('Total Return vs Confidence Threshold')
        ax1.grid(True)
        
        # Plot 2: Number of Trades vs Threshold
        ax2.plot(df['threshold'], df['num_trades'], 'go-')
        ax2.set_ylabel('Number of Trades')
        ax2.set_title('Number of Trades vs Confidence Threshold')
        ax2.grid(True)
        
        # Plot 3: Win Rate vs Threshold
        ax3.plot(df['threshold'], df['win_rate'], 'ro-')
        ax3.set_ylabel('Win Rate (%)')
        ax3.set_xlabel('Confidence Threshold')
        ax3.set_title('Win Rate vs Confidence Threshold')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print results
        print("\nThreshold Analysis Results:")
        print(df.to_string(index=False))
        
        # Find optimal threshold
        if not df.empty and not df['total_return'].isna().all():
            max_return_idx = df['total_return'].idxmax()
            optimal_threshold = df.iloc[max_return_idx]['threshold']
            max_return = df.iloc[max_return_idx]['total_return']
            
            print(f"\nOptimal confidence threshold: {optimal_threshold}")
            print(f"Maximum return: {max_return:.2f}%")
        else:
            print("\nNo valid results to determine optimal threshold")
        
        return df

    def run_multiple_simulations(self, num_sims=5, duration_minutes=60, sequence_length=100):
        """
        Run multiple simulations with different market conditions
        
        Parameters:
        -----------
        num_sims : int
            Number of simulations to run
        duration_minutes : int
            Duration of each simulation in minutes
        sequence_length : int
            Length of order sequence for each prediction
        """
        print(f"\nRunning {num_sims} simulations with different market conditions...")
        
        results = []
        
        # Define different market conditions
        conditions = [
            {'trend': 0.0, 'volatility': 0.02, 'name': 'Neutral Market'},
            {'trend': 0.01, 'volatility': 0.02, 'name': 'Bullish Market'},
            {'trend': -0.01, 'volatility': 0.02, 'name': 'Bearish Market'},
            {'trend': 0.0, 'volatility': 0.04, 'name': 'High Volatility'},
            {'trend': 0.0, 'volatility': 0.01, 'name': 'Low Volatility'}
        ]
        
        # Run simulations for each condition
        for i, condition in enumerate(conditions[:num_sims]):
            print(f"\nSimulation {i+1}/{num_sims}: {condition['name']}")
            print(f"Trend: {condition['trend']}, Volatility: {condition['volatility']}")
            
            result = self.run_simulation(
                duration_minutes=duration_minutes, 
                sequence_length=sequence_length,
                trend=condition['trend'],
                volatility=condition['volatility']
            )
            
            results.append({
                'name': condition['name'],
                'trend': condition['trend'],
                'volatility': condition['volatility'],
                'initial_value': result['initial_value'],
                'final_value': result['final_value'],
                'total_return': result['total_return'],
                'num_trades': result['num_trades'],
                'price_change': result['price_change']
            })
        
        # Create summary DataFrame
        summary = pd.DataFrame(results)
        
        print("\nSummary of All Simulations:")
        print(summary.to_string(index=False))
        
        # Create a summary plot
        plt.figure(figsize=(12, 8))
        
        # Plot returns vs market conditions
        plt.barh(summary['name'], summary['total_return'], color='blue', alpha=0.7)
        plt.xlabel('Total Return (%)')
        plt.title('Strategy Performance Across Different Market Conditions')
        plt.axvline(x=0, color='red', linestyle='--')
        plt.grid(True, axis='x')
        
        plt.tight_layout()
        plt.show()
        
        # Plot correlation between price change and strategy return
        plt.figure(figsize=(10, 6))
        plt.scatter(summary['price_change'], summary['total_return'], s=100, alpha=0.7)
        
        for i, row in summary.iterrows():
            plt.annotate(row['name'], (row['price_change'], row['total_return']),
                         xytext=(10, 5), textcoords='offset points')
        
        plt.xlabel('Market Price Change (%)')
        plt.ylabel('Strategy Return (%)')
        plt.title('Strategy Return vs Market Price Change')
        plt.grid(True)
        
        # Add trend line
        if len(summary) > 1:
            z = np.polyfit(summary['price_change'], summary['total_return'], 1)
            p = np.poly1d(z)
            plt.plot(summary['price_change'], p(summary['price_change']), "r--")
        
        plt.tight_layout()
        plt.show()
        
        return summary


# Main execution
def run_experiment():
    # Create and train strategy
    strategy = StockPricePredictionStrategy(threshold=0.6, device=device)
    
    # Train model with simulated data
    print("\n========== TRAINING MODEL ==========")
    X_test, y_test, all_probs = strategy.train_model(
        num_samples=3000,
        batch_size=32,
        num_epochs=30,
        learning_rate=0.001,
        weight_decay=1e-5
    )
    
    # Run simulation
    print("\n========== RUNNING SIMULATION ==========")
    strategy.run_simulation(duration_minutes=60)
    
    # Analyze threshold impact
    print("\n========== ANALYZING THRESHOLD IMPACT ==========")
    threshold_analysis = strategy.analyze_threshold_impact()
    
    # Run multiple simulations with different market conditions
    print("\n========== TESTING DIFFERENT MARKET CONDITIONS ==========")
    market_analysis = strategy.run_multiple_simulations(num_sims=5)
    
    return strategy

if __name__ == "__main__":
    print("A-CNN+ Stock Price Prediction Strategy")
    print("Based on: 'Encoding of high-frequency order information and prediction of short-term stock price by deep learning'")
    print("===============================================================")
    
    strategy = run_experiment()