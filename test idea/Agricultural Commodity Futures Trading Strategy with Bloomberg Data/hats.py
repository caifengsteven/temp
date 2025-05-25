import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm

# Set random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Generate Simulated Data

def generate_price_data(num_companies=500, num_days=1000, volatility=0.02):
    """Generate simulated price data for multiple companies."""
    # Initialize price data
    initial_prices = np.random.uniform(10, 100, num_companies)
    prices = np.zeros((num_companies, num_days))
    prices[:, 0] = initial_prices
    
    # Generate daily returns with correlation
    correlation_matrix = np.random.uniform(-0.2, 0.8, (num_companies, num_companies))
    np.fill_diagonal(correlation_matrix, 1)
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Make it symmetric
    
    # Ensure positive semi-definite
    eigenvalues = np.linalg.eigvalsh(correlation_matrix)
    if np.any(eigenvalues < 0):
        min_eig = np.min(eigenvalues)
        correlation_matrix += (abs(min_eig) + 0.01) * np.eye(num_companies)
    
    # Cholesky decomposition for correlated random variables
    L = np.linalg.cholesky(correlation_matrix)
    
    # Generate returns for all days at once
    mean_returns = np.random.uniform(-0.0005, 0.0015, num_companies)
    uncorrelated_returns = np.random.normal(0, volatility, (num_companies, num_days-1))
    daily_returns = mean_returns[:, np.newaxis] + np.dot(L, uncorrelated_returns)
    
    # Calculate prices
    for day in range(1, num_days):
        prices[:, day] = prices[:, day-1] * (1 + daily_returns[:, day-1])
    
    return prices

def generate_market_indexes(prices, num_indexes=5):
    """Create market indexes from the price data."""
    num_companies = prices.shape[0]
    num_days = prices.shape[1]
    
    # Create random company assignments to indexes
    index_constituents = []
    for _ in range(num_indexes):
        # Each index consists of a random subset of companies
        index_size = random.randint(30, 100)
        constituents = np.random.choice(num_companies, size=index_size, replace=False)
        index_constituents.append(constituents)
    
    # Calculate index values (weighted average of constituent prices)
    index_prices = np.zeros((num_indexes, num_days))
    for idx, constituents in enumerate(index_constituents):
        weights = np.random.dirichlet(np.ones(len(constituents))) * len(constituents)
        index_prices[idx] = np.dot(weights, prices[constituents]) / np.sum(weights)
    
    return index_prices, index_constituents

def generate_company_relations(num_companies=500, num_relation_types=10):
    """Generate simulated company relation data."""
    # Create a graph for each relation type
    relations = {}
    relation_types = [f"relation_{i}" for i in range(num_relation_types)]
    
    for relation_type in relation_types:
        G = nx.DiGraph()
        G.add_nodes_from(range(num_companies))
        
        # Determine relation density (different for each relation type)
        if relation_type.endswith('_0') or relation_type.endswith('_1'):  # Industry/Subsidiary type relations
            edge_prob = 0.02  # More sparse but meaningful relations
        elif relation_type.endswith('_2') or relation_type.endswith('_3'):  # Geographic type relations
            edge_prob = 0.1  # More dense but less meaningful
        else:
            edge_prob = 0.005  # Very sparse relations
        
        # Create edges
        for i in range(num_companies):
            for j in range(num_companies):
                if i != j and random.random() < edge_prob:
                    G.add_edge(i, j)
        
        relations[relation_type] = G
    
    return relations, relation_types

def prepare_price_features(prices, seq_length=50):
    """Prepare price change features for model input."""
    num_companies, num_days = prices.shape
    
    # Calculate price changes
    price_changes = np.zeros_like(prices)
    price_changes[:, 1:] = (prices[:, 1:] - prices[:, :-1]) / prices[:, :-1]
    
    # Prepare sequences
    X = []
    y = []
    timestamps = []
    
    for day in range(seq_length, num_days-1):
        X.append(price_changes[:, day-seq_length:day])
        
        # Create labels: -1 (down), 0 (neutral), 1 (up)
        next_day_change = price_changes[:, day]
        thresholds = np.percentile(next_day_change, [33, 67])
        labels = np.zeros(num_companies)
        labels[next_day_change < thresholds[0]] = -1
        labels[next_day_change > thresholds[1]] = 1
        y.append(labels)
        
        timestamps.append(day)
    
    return np.array(X), np.array(y), timestamps

def prepare_index_labels(index_prices, timestamps):
    """Prepare labels for market index prediction."""
    num_indexes, num_days = index_prices.shape
    
    # Calculate price changes
    index_changes = np.zeros_like(index_prices)
    index_changes[:, 1:] = (index_prices[:, 1:] - index_prices[:, :-1]) / index_prices[:, :-1]
    
    # Create labels: -1 (down), 0 (neutral), 1 (up)
    y_index = []
    for day in timestamps:
        next_day_change = index_changes[:, day]
        thresholds = np.percentile(next_day_change, [33, 67])
        labels = np.zeros(num_indexes)
        labels[next_day_change < thresholds[0]] = -1
        labels[next_day_change > thresholds[1]] = 1
        y_index.append(labels)
    
    return np.array(y_index)

def prepare_relation_data(relations, num_companies):
    """Convert relation graphs to adjacency matrices."""
    adj_matrices = {}
    for rel_type, G in relations.items():
        adj_matrix = nx.to_numpy_array(G)
        adj_matrices[rel_type] = torch.FloatTensor(adj_matrix)
    
    return adj_matrices

# HATS Model Implementation

class FeatureExtractor(nn.Module):
    """LSTM/GRU module for time series feature extraction."""
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=1, dropout=0.2, model_type='gru'):
        super(FeatureExtractor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model_type = model_type
        
        if model_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
            
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim)
        output, _ = self.rnn(x)
        # Return only the last time step
        return output[:, -1, :]

class RelationalAttention(nn.Module):
    """Hierarchical attention mechanism for relation-based information aggregation."""
    def __init__(self, node_dim, relation_types, hidden_dim=32):
        super(RelationalAttention, self).__init__()
        self.relation_types = relation_types
        self.relation_embeddings = nn.Embedding(len(relation_types), hidden_dim)
        
        # State attention layer
        self.state_query = nn.Linear(node_dim * 2 + hidden_dim, 1)
        
        # Relation attention layer
        self.relation_query = nn.Linear(node_dim * 2 + hidden_dim, 1)
        
    def forward(self, node_features, adj_matrices):
        batch_size, num_nodes, node_dim = node_features.size()
        
        # Initialize relation representations
        relation_aggregated = {}
        
        for i, rel_type in enumerate(self.relation_types):
            if rel_type in adj_matrices:
                adj_matrix = adj_matrices[rel_type].to(node_features.device)
                
                # Get relation embedding
                rel_embedding = self.relation_embeddings(torch.tensor([i], device=node_features.device))
                rel_embedding_expanded = rel_embedding.expand(batch_size, num_nodes, -1)
                
                # For each node, get features of all neighbors
                neighbor_features = torch.matmul(adj_matrix, node_features)
                
                # State attention for each node and its neighbors
                # Concatenate node features, neighbor features, and relation embedding
                concat_features = torch.cat([
                    node_features,
                    neighbor_features,
                    rel_embedding_expanded
                ], dim=2)
                
                # Calculate state attention scores
                state_scores = self.state_query(concat_features)
                state_weights = F.softmax(state_scores, dim=1)
                
                # Weighted aggregation of neighbor features
                relation_aggregated[rel_type] = torch.sum(state_weights * neighbor_features, dim=1)
            else:
                # If no edges for this relation type, use a zero vector
                relation_aggregated[rel_type] = torch.zeros(batch_size, node_dim, device=node_features.device)
        
        # Relation attention layer
        relation_scores = {}
        for i, rel_type in enumerate(self.relation_types):
            rel_embedding = self.relation_embeddings(torch.tensor([i], device=node_features.device))
            rel_embedding_expanded = rel_embedding.expand(batch_size, -1)
            
            # Concatenate node features, relation features, and relation embedding
            concat_features = torch.cat([
                node_features.mean(dim=1),
                relation_aggregated[rel_type],
                rel_embedding_expanded
            ], dim=1)
            
            # Calculate relation attention score
            relation_scores[rel_type] = self.relation_query(concat_features)
        
        # Normalize relation scores with softmax
        relation_score_tensor = torch.cat([relation_scores[r] for r in self.relation_types], dim=1)
        relation_weights = F.softmax(relation_score_tensor, dim=1)
        
        # Weighted aggregation of relation features
        relation_features = torch.stack([relation_aggregated[r] for r in self.relation_types], dim=1)
        aggregated_features = torch.bmm(relation_weights.unsqueeze(1), relation_features).squeeze(1)
        
        # Reshape to match original node features
        aggregated_features = aggregated_features.unsqueeze(1).expand(-1, num_nodes, -1)
        
        # Add to original node features
        updated_features = node_features + aggregated_features
        
        return updated_features, {r: relation_weights[:, i].mean().item() for i, r in enumerate(self.relation_types)}

class HATS(nn.Module):
    """Hierarchical Attention Network for Stock Movement Prediction."""
    def __init__(self, input_dim, hidden_dim, num_classes, relation_types, num_layers=1, dropout=0.2):
        super(HATS, self).__init__()
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim, num_layers, dropout)
        self.relational_attention = RelationalAttention(hidden_dim, relation_types, hidden_dim//2)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, adj_matrices):
        # Reshape input: (batch_size, num_companies, seq_length) -> (batch_size * num_companies, seq_length, 1)
        batch_size, num_companies, seq_length = x.size()
        x_reshaped = x.view(batch_size * num_companies, seq_length, 1)
        
        # Feature extraction
        node_features = self.feature_extractor(x_reshaped)
        
        # Reshape back: (batch_size * num_companies, hidden_dim) -> (batch_size, num_companies, hidden_dim)
        node_features = node_features.view(batch_size, num_companies, -1)
        
        # Relational modeling
        updated_features, relation_weights = self.relational_attention(node_features, adj_matrices)
        
        # Reshape for classification: (batch_size, num_companies, hidden_dim) -> (batch_size * num_companies, hidden_dim)
        updated_features_flat = updated_features.view(batch_size * num_companies, -1)
        
        # Task-specific prediction
        logits = self.classifier(updated_features_flat)
        
        # Reshape logits: (batch_size * num_companies, num_classes) -> (batch_size, num_companies, num_classes)
        logits = logits.view(batch_size, num_companies, -1)
        
        return logits, relation_weights

# Training and Evaluation Functions

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (x, y, _) in enumerate(train_loader):
        optimizer.zero_grad()
        
        x = x.to(device)
        y = y.to(device)
        
        # Get adjacency matrices
        adj_matrices = {rel_type: adj.to(device) for rel_type, adj in train_adj_matrices.items()}
        
        logits, _ = model(x, adj_matrices)
        
        # Reshape for loss calculation
        logits_flat = logits.view(-1, logits.size(-1))
        y_flat = y.view(-1)
        
        loss = criterion(logits_flat, y_flat)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / (batch_idx + 1)

def evaluate(model, data_loader, criterion, device, adj_matrices):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (x, y, _) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)
            
            # Get adjacency matrices
            adj_matrices_device = {rel_type: adj.to(device) for rel_type, adj in adj_matrices.items()}
            
            logits, _ = model(x, adj_matrices_device)
            
            # Reshape for loss calculation
            logits_flat = logits.view(-1, logits.size(-1))
            y_flat = y.view(-1)
            
            loss = criterion(logits_flat, y_flat)
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits_flat, dim=1).cpu().numpy()
            labels = y_flat.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # Filter out padded values
    valid_indices = [i for i, label in enumerate(all_labels) if label != -100]
    filtered_preds = [all_preds[i] for i in valid_indices]
    filtered_labels = [all_labels[i] for i in valid_indices]
    
    accuracy = accuracy_score(filtered_labels, filtered_preds)
    f1 = f1_score(filtered_labels, filtered_preds, average='macro')
    
    return total_loss / (batch_idx + 1), accuracy, f1

def prepare_data_loaders(X, y, batch_size=32):
    """Prepare data loaders for training and evaluation."""
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Create dataset
    dataset = [(X_tensor[i], y_tensor[i], i) for i in range(len(X_tensor))]
    
    # Split into batches
    batches = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]
    
    # Create data loaders
    data_loader = []
    
    for batch in batches:
        batch_X = torch.stack([item[0] for item in batch])
        batch_y = torch.stack([item[1] for item in batch])
        batch_idx = [item[2] for item in batch]
        data_loader.append((batch_X, batch_y, batch_idx))
    
    return data_loader

def train_and_evaluate(model, train_loader, val_loader, train_adj_matrices, val_adj_matrices, num_epochs=50, learning_rate=0.001):
    """Train and evaluate the model."""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    best_val_f1 = 0
    best_model = None
    
    train_losses = []
    val_losses = []
    val_f1s = []
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device, val_adj_matrices)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model.state_dict().copy()
    
    # Load best model
    model.load_state_dict(best_model)
    
    return model, train_losses, val_losses, val_f1s

# Trading Strategy Functions

def create_trading_portfolio(model, test_loader, test_adj_matrices, device, neutralized=True):
    """Create a trading portfolio based on model predictions."""
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for x, _, _ in test_loader:
            x = x.to(device)
            
            # Get adjacency matrices
            adj_matrices = {rel_type: adj.to(device) for rel_type, adj in test_adj_matrices.items()}
            
            logits, _ = model(x, adj_matrices)
            probs = F.softmax(logits, dim=2)
            
            # Store company probabilities for each day
            all_probs.append(probs.cpu().numpy())
    
    # Create a portfolio for each test day
    daily_weights = []
    
    for day_probs in all_probs:
        # Get probabilities for this day (across all companies)
        day_probs = day_probs.reshape(-1, 3)  # (num_companies, 3)
        
        if neutralized:
            # Long the top 15% stocks with highest up probability
            # Short the top 15% stocks with highest down probability
            up_probs = day_probs[:, 2]  # Assuming class 2 is "up"
            down_probs = day_probs[:, 0]  # Assuming class 0 is "down"
            
            num_stocks = len(up_probs)
            num_long = int(0.15 * num_stocks)
            num_short = int(0.15 * num_stocks)
            
            long_indices = np.argsort(up_probs)[-num_long:]
            short_indices = np.argsort(down_probs)[-num_short:]
            
            # Create portfolio weights
            weights = np.zeros(num_stocks)
            weights[long_indices] = 1.0 / num_long
            weights[short_indices] = -1.0 / num_short
        else:
            # Simple strategy: weight proportional to up probability minus down probability
            weights = (day_probs[:, 2] - day_probs[:, 0])
            weights = weights / np.sum(np.abs(weights))
        
        daily_weights.append(weights)
    
    return np.array(daily_weights)

def backtest_strategy(daily_weights, returns, risk_free_rate=0.01/252):
    """Backtest a trading strategy with daily rebalancing."""
    # daily_weights shape: (num_days, num_companies)
    # returns shape: (num_companies, num_days)
    
    # Ensure returns has the right shape (num_days, num_companies)
    returns_T = returns.T
    
    # Ensure shapes match (use min in case we have more predictions than returns)
    min_days = min(daily_weights.shape[0], returns_T.shape[0])
    
    daily_weights = daily_weights[:min_days]
    returns_T = returns_T[:min_days]
    
    # Calculate daily portfolio returns
    portfolio_returns = np.sum(daily_weights * returns_T, axis=1)
    
    # Calculate performance metrics
    cumulative_returns = np.cumprod(1 + portfolio_returns) - 1
    annualized_return = np.mean(portfolio_returns) * 252
    annualized_vol = np.std(portfolio_returns) * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0
    
    # Calculate drawdown
    cumulative = np.cumprod(1 + portfolio_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative / running_max) - 1
    max_drawdown = np.min(drawdown)
    
    return {
        'portfolio_returns': portfolio_returns,
        'cumulative_returns': cumulative_returns,
        'annualized_return': annualized_return,
        'annualized_vol': annualized_vol,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

# Main Execution

# Parameters
num_companies = 200
num_days = 1500
seq_length = 50
num_relation_types = 10
hidden_dim = 64
num_epochs = 20
batch_size = 16
learning_rate = 0.001

# Generate data
print("Generating simulated data...")
prices = generate_price_data(num_companies, num_days)
index_prices, index_constituents = generate_market_indexes(prices)
relations, relation_types = generate_company_relations(num_companies, num_relation_types)

# Prepare features and labels
X, y, timestamps = prepare_price_features(prices, seq_length)
y_index = prepare_index_labels(index_prices, timestamps)

# Convert relation data to adjacency matrices
adj_matrices = prepare_relation_data(relations, num_companies)

# Convert labels from [-1,0,1] to [0,1,2] for classification
y_classes = y + 1

# Split data for training, validation, and testing
train_cutoff = int(0.7 * len(X))
val_cutoff = int(0.85 * len(X))

X_train, y_train = X[:train_cutoff], y_classes[:train_cutoff]
X_val, y_val = X[train_cutoff:val_cutoff], y_classes[train_cutoff:val_cutoff]
X_test, y_test = X[val_cutoff:], y_classes[val_cutoff:]

# Calculate returns for backtesting
price_changes = np.diff(prices, axis=1) / prices[:, :-1]
test_returns = price_changes[:, val_cutoff+seq_length-1:-1]  # Align with test predictions

# Prepare data loaders
train_loader = prepare_data_loaders(X_train, y_train, batch_size)
val_loader = prepare_data_loaders(X_val, y_val, batch_size)
test_loader = prepare_data_loaders(X_test, y_test, batch_size=1)  # Batch size 1 for testing

# Create separate adjacency matrices for each dataset
train_adj_matrices = adj_matrices
val_adj_matrices = adj_matrices
test_adj_matrices = adj_matrices

# Initialize and train the HATS model
print("Training HATS model...")
model = HATS(input_dim=1, hidden_dim=hidden_dim, num_classes=3, relation_types=relation_types).to(device)
trained_model, train_losses, val_losses, val_f1s = train_and_evaluate(
    model, train_loader, val_loader, train_adj_matrices, val_adj_matrices, num_epochs, learning_rate
)

# Evaluate on test set
test_loss, test_acc, test_f1 = evaluate(trained_model, test_loader, nn.CrossEntropyLoss(ignore_index=-100), device, test_adj_matrices)
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")

# Implement trading strategy
print("Backtesting trading strategy...")
portfolio_weights = create_trading_portfolio(trained_model, test_loader, test_adj_matrices, device)
backtest_results = backtest_strategy(portfolio_weights, test_returns)

print(f"Trading Strategy Results:")
print(f"Annualized Return: {backtest_results['annualized_return']*100:.2f}%")
print(f"Annualized Volatility: {backtest_results['annualized_vol']*100:.2f}%")
print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
print(f"Maximum Drawdown: {backtest_results['max_drawdown']*100:.2f}%")

# Compare with baseline LSTM model

# LSTM Baseline
class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_classes=3, num_layers=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # Reshape input: (batch_size, num_companies, seq_length) -> (batch_size * num_companies, seq_length, 1)
        batch_size, num_companies, seq_length = x.size()
        x_reshaped = x.view(batch_size * num_companies, seq_length, 1)
        
        # LSTM forward
        lstm_out, _ = self.lstm(x_reshaped)
        lstm_out = lstm_out[:, -1, :]  # Take last hidden state
        
        # Classification
        logits = self.fc(lstm_out)
        
        # Reshape logits: (batch_size * num_companies, num_classes) -> (batch_size, num_companies, num_classes)
        logits = logits.view(batch_size, num_companies, -1)
        
        return logits

def train_lstm_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (x, y, _) in enumerate(train_loader):
        optimizer.zero_grad()
        
        x = x.to(device)
        y = y.to(device)
        
        logits = model(x)
        
        # Reshape for loss calculation
        logits_flat = logits.view(-1, logits.size(-1))
        y_flat = y.view(-1)
        
        loss = criterion(logits_flat, y_flat)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / (batch_idx + 1)

def evaluate_lstm(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (x, y, _) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)
            
            logits = model(x)
            
            # Reshape for loss calculation
            logits_flat = logits.view(-1, logits.size(-1))
            y_flat = y.view(-1)
            
            loss = criterion(logits_flat, y_flat)
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits_flat, dim=1).cpu().numpy()
            labels = y_flat.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    # Filter out padded values
    valid_indices = [i for i, label in enumerate(all_labels) if label != -100]
    filtered_preds = [all_preds[i] for i in valid_indices]
    filtered_labels = [all_labels[i] for i in valid_indices]
    
    accuracy = accuracy_score(filtered_labels, filtered_preds)
    f1 = f1_score(filtered_labels, filtered_preds, average='macro')
    
    return total_loss / (batch_idx + 1), accuracy, f1

def create_lstm_trading_portfolio(model, test_loader, device, neutralized=True):
    """Create a trading portfolio based on LSTM model predictions."""
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for x, _, _ in test_loader:
            x = x.to(device)
            
            logits = model(x)
            probs = F.softmax(logits, dim=2)
            
            # Store company probabilities for each day
            all_probs.append(probs.cpu().numpy())
    
    # Create a portfolio for each test day
    daily_weights = []
    
    for day_probs in all_probs:
        # Get probabilities for this day (across all companies)
        day_probs = day_probs.reshape(-1, 3)  # (num_companies, 3)
        
        if neutralized:
            # Long the top 15% stocks with highest up probability
            # Short the top 15% stocks with highest down probability
            up_probs = day_probs[:, 2]  # Assuming class 2 is "up"
            down_probs = day_probs[:, 0]  # Assuming class 0 is "down"
            
            num_stocks = len(up_probs)
            num_long = int(0.15 * num_stocks)
            num_short = int(0.15 * num_stocks)
            
            long_indices = np.argsort(up_probs)[-num_long:]
            short_indices = np.argsort(down_probs)[-num_short:]
            
            # Create portfolio weights
            weights = np.zeros(num_stocks)
            weights[long_indices] = 1.0 / num_long
            weights[short_indices] = -1.0 / num_short
        else:
            # Simple strategy: weight proportional to up probability minus down probability
            weights = (day_probs[:, 2] - day_probs[:, 0])
            weights = weights / np.sum(np.abs(weights))
        
        daily_weights.append(weights)
    
    return np.array(daily_weights)

print("Training LSTM baseline model...")
lstm_model = LSTMModel(input_dim=1, hidden_dim=hidden_dim, num_classes=3).to(device)
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate)
lstm_criterion = nn.CrossEntropyLoss(ignore_index=-100)

lstm_train_losses = []
lstm_val_losses = []
lstm_val_f1s = []

for epoch in range(num_epochs):
    lstm_train_loss = train_lstm_epoch(lstm_model, train_loader, lstm_optimizer, lstm_criterion, device)
    lstm_val_loss, lstm_val_acc, lstm_val_f1 = evaluate_lstm(lstm_model, val_loader, lstm_criterion, device)
    
    lstm_train_losses.append(lstm_train_loss)
    lstm_val_losses.append(lstm_val_loss)
    lstm_val_f1s.append(lstm_val_f1)
    
    print(f"LSTM Epoch {epoch+1}/{num_epochs} - Train Loss: {lstm_train_loss:.4f}, Val Loss: {lstm_val_loss:.4f}, Val Acc: {lstm_val_acc:.4f}, Val F1: {lstm_val_f1:.4f}")

# Evaluate LSTM on test set
lstm_test_loss, lstm_test_acc, lstm_test_f1 = evaluate_lstm(lstm_model, test_loader, lstm_criterion, device)
print(f"LSTM Test Loss: {lstm_test_loss:.4f}, Test Acc: {lstm_test_acc:.4f}, Test F1: {lstm_test_f1:.4f}")

# LSTM Trading Strategy
lstm_portfolio_weights = create_lstm_trading_portfolio(lstm_model, test_loader, device)
lstm_backtest = backtest_strategy(lstm_portfolio_weights, test_returns)

print(f"LSTM Trading Strategy Results:")
print(f"Annualized Return: {lstm_backtest['annualized_return']*100:.2f}%")
print(f"Annualized Volatility: {lstm_backtest['annualized_vol']*100:.2f}%")
print(f"Sharpe Ratio: {lstm_backtest['sharpe_ratio']:.2f}")
print(f"Maximum Drawdown: {lstm_backtest['max_drawdown']*100:.2f}%")

# Plot results
plt.figure(figsize=(15, 12))

# Plot 1: Training and Validation Loss
plt.subplot(2, 2, 1)
plt.plot(train_losses, label='HATS Train Loss')
plt.plot(val_losses, label='HATS Val Loss')
plt.plot(lstm_train_losses, label='LSTM Train Loss')
plt.plot(lstm_val_losses, label='LSTM Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot 2: Validation F1 Score
plt.subplot(2, 2, 2)
plt.plot(val_f1s, label='HATS F1')
plt.plot(lstm_val_f1s, label='LSTM F1')
plt.title('Validation F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()

# Plot 3: Cumulative Strategy Returns
plt.subplot(2, 2, 3)
plt.plot(backtest_results['cumulative_returns'], label='HATS Strategy')
plt.plot(lstm_backtest['cumulative_returns'], label='LSTM Strategy')
plt.title('Cumulative Strategy Returns')
plt.xlabel('Trading Day')
plt.ylabel('Cumulative Return')
plt.legend()

# Plot 4: Relation Attention Weights
plt.subplot(2, 2, 4)
# Get relation weights from the model
model.eval()
with torch.no_grad():
    # Get first batch from test loader
    x, _, _ = test_loader[0]
    x = x.to(device)
    
    # Get adjacency matrices
    adj_matrices_device = {rel_type: adj.to(device) for rel_type, adj in test_adj_matrices.items()}
    
    # Forward pass to get relation weights
    _, relation_weights = model(x, adj_matrices_device)

# Plot relation weights
rel_names = [f"Relation {i}" for i in range(len(relation_types))]
rel_weight_values = [relation_weights[r] for r in relation_types]
plt.bar(rel_names, rel_weight_values)
plt.title('Relation Attention Weights')
plt.xlabel('Relation Type')
plt.ylabel('Weight')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('hats_results.png')
plt.show()

print("Finished! Results saved to 'hats_results.png'")