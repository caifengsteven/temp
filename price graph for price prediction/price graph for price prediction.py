#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Price Graph Strategy Implementation

Based on the paper:
"Price graphs: Utilizing the structural information of financial time series for stock prediction"
by Junran Wu, Ke Xu, Xueyuan Chen, Shangzhe Li, Jichang Zhao
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
import warnings
from typing import List, Dict, Tuple, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ignore warnings
warnings.filterwarnings("ignore")

# Check if cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


class PriceGraphDataset:
    """
    Dataset class for creating visibility graphs from financial time series data
    and extracting structural information.
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 window_size: int = 20,
                 target_horizon: int = 1,
                 embedding_dim: int = 32):
        """
        Initialize the dataset.
        
        Args:
            data: DataFrame with columns ['open', 'high', 'low', 'close', 'volume', 'amount']
            window_size: Size of the lookback window
            target_horizon: Number of days ahead to predict
            embedding_dim: Dimension of node embeddings
        """
        self.data = data
        self.window_size = window_size
        self.target_horizon = target_horizon
        self.embedding_dim = embedding_dim
        
        # Create features and labels
        self.X_graph_embeddings = []
        self.X_node_weights = []
        self.y = []
        
        # Create visibility graphs and extract features
        self._create_visibility_graphs()
        
    def _create_visibility_graph(self, series: np.ndarray) -> nx.Graph:
        """
        Create a visibility graph from a time series.
        
        Args:
            series: 1D numpy array of time series values
            
        Returns:
            Visibility graph as networkx Graph
        """
        n = len(series)
        G = nx.Graph()
        
        # Add nodes
        G.add_nodes_from(range(n))
        
        # Add edges according to visibility criterion
        for i in range(n):
            for j in range(i+1, n):
                # Check visibility between nodes i and j
                visible = True
                for k in range(i+1, j):
                    # Check if any intermediate node blocks visibility
                    if series[k] >= series[i] + (series[j] - series[i]) * (k - i) / (j - i):
                        visible = False
                        break
                
                if visible:
                    G.add_edge(i, j)
        
        return G
    
    def _calculate_node_weights(self, G: nx.Graph, l: int = 2) -> np.ndarray:
        """
        Calculate Collective Influence (CI) for nodes in the graph.
        
        Args:
            G: Networkx graph
            l: Radius parameter for CI calculation
            
        Returns:
            Array of CI values for each node
        """
        n = G.number_of_nodes()
        ci_values = np.zeros(n)
        
        try:
            for i in range(n):
                # Get nodes at distance l from node i
                frontier_nodes = []
                for node in G.nodes():
                    try:
                        if nx.shortest_path_length(G, i, node) == l:
                            frontier_nodes.append(node)
                    except:
                        # No path between i and node or other error
                        continue
                
                # Calculate CI for node i
                ci_i = max(0, G.degree(i) - 1)  # Avoid negative values
                sum_term = 0
                for j in frontier_nodes:
                    sum_term += max(0, G.degree(j) - 1)  # Avoid negative values
                
                ci_values[i] = ci_i * sum_term
        except Exception as e:
            logger.warning(f"Error calculating CI values: {e}, using degree as fallback")
            # Fallback to using node degrees
            for i in range(n):
                ci_values[i] = G.degree(i)
        
        return ci_values
    
    def _get_graph_embeddings(self, G: nx.Graph) -> np.ndarray:
        """
        Get simple, consistent embeddings for nodes in the graph.
        
        Args:
            G: Networkx graph
            
        Returns:
            Node embeddings matrix of shape (n_nodes, embedding_dim)
        """
        n = G.number_of_nodes()
        
        # Create a fixed-size embedding matrix
        embeddings = np.zeros((n, self.embedding_dim))
        
        # Extract basic node features
        for i in range(n):
            if i >= G.number_of_nodes():
                # Safety check in case n is larger than actual nodes
                continue
                
            # Node degree (normalized)
            degree = G.degree(i) / max(1, n)
            
            # Node clustering coefficient
            try:
                clustering = nx.clustering(G, i)
            except:
                clustering = 0
            
            # Degree centrality
            try:
                centrality = nx.degree_centrality(G)[i]
            except:
                centrality = 0
                
            # Closeness centrality (or approximation)
            try:
                closeness = nx.closeness_centrality(G, i)
            except:
                closeness = 0
                
            # Fill the first few dimensions with these features
            embeddings[i, 0] = degree
            embeddings[i, 1] = clustering
            embeddings[i, 2] = centrality
            embeddings[i, 3] = closeness
            
            # Position in sequence (normalized)
            embeddings[i, 4] = i / max(1, n)
            
            # Fill remaining dimensions with random values seeded by node features
            np.random.seed(int(1000 * (degree + clustering + centrality + i)))
            embeddings[i, 5:] = np.random.randn(self.embedding_dim - 5) * 0.1
        
        return embeddings
    
    def _create_visibility_graphs(self):
        """Create visibility graphs for each window in the data."""
        # Get all required features
        features = ['open', 'high', 'low', 'close', 'volume', 'amount']
        
        logger.info("Creating visibility graphs and extracting structural features...")
        for i in tqdm(range(self.window_size, len(self.data) - self.target_horizon)):
            window_data = self.data.iloc[i - self.window_size:i]
            
            # Create graphs and extract information for each feature
            window_embeddings = []
            window_weights = []
            
            for feature in features:
                # Create visibility graph
                series = window_data[feature].values
                G = self._create_visibility_graph(series)
                
                # Get graph embeddings - make sure shape is (window_size, embedding_dim)
                embeddings = self._get_graph_embeddings(G)
                # Make sure embeddings has shape (window_size, embedding_dim)
                if embeddings.shape[0] != self.window_size:
                    # Pad or truncate to ensure consistent shape
                    if embeddings.shape[0] < self.window_size:
                        padding = np.zeros((self.window_size - embeddings.shape[0], self.embedding_dim))
                        embeddings = np.vstack([embeddings, padding])
                    else:
                        embeddings = embeddings[:self.window_size]
                
                window_embeddings.append(embeddings)
                
                # Calculate node weights (CI)
                weights = self._calculate_node_weights(G)
                # Make sure weights has shape (window_size,)
                if len(weights) != self.window_size:
                    # Pad or truncate to ensure consistent shape
                    if len(weights) < self.window_size:
                        padding = np.zeros(self.window_size - len(weights))
                        weights = np.concatenate([weights, padding])
                    else:
                        weights = weights[:self.window_size]
                
                window_weights.append(weights)
            
            try:
                # Stack embeddings and weights with clearly defined shapes
                # Each embedding has shape (window_size, embedding_dim)
                # After stacking, shape should be (n_features, window_size, embedding_dim)
                stacked_embeddings = np.stack(window_embeddings)
                
                # Each weight vector has shape (window_size,)
                # After stacking, shape should be (n_features, window_size)
                stacked_weights = np.stack(window_weights)
                
                # Store in our lists
                self.X_graph_embeddings.append(stacked_embeddings)
                self.X_node_weights.append(stacked_weights)
                
                # Create target: 1 if close price increases, 0 otherwise
                target_price = self.data.iloc[i + self.target_horizon]['close']
                current_price = self.data.iloc[i]['close']
                self.y.append(1 if target_price > current_price else 0)
            except Exception as e:
                logger.warning(f"Error processing window {i}: {e}, skipping this window")
                continue
        
        # Convert to arrays if we have any data
        if len(self.y) > 0:
            self.X_graph_embeddings = np.array(self.X_graph_embeddings)
            self.X_node_weights = np.array(self.X_node_weights)
            self.y = np.array(self.y)
            
            logger.info(f"Created {len(self.y)} samples with embeddings shape {self.X_graph_embeddings.shape} "
                       f"and weights shape {self.X_node_weights.shape}")
        else:
            logger.error("No valid samples were generated. Please check your data.")
    
    def split_data(self, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple:
        """
        Split data into train, validation, and test sets.
        
        Args:
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        if len(self.y) == 0:
            logger.error("No data to split. Dataset is empty.")
            return None, None, None
            
        n_samples = len(self.y)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        # Create train set
        train_data = {
            'embeddings': self.X_graph_embeddings[:train_size],
            'weights': self.X_node_weights[:train_size],
            'labels': self.y[:train_size]
        }
        
        # Create validation set
        val_data = {
            'embeddings': self.X_graph_embeddings[train_size:train_size + val_size],
            'weights': self.X_node_weights[train_size:train_size + val_size],
            'labels': self.y[train_size:train_size + val_size]
        }
        
        # Create test set
        test_data = {
            'embeddings': self.X_graph_embeddings[train_size + val_size:],
            'weights': self.X_node_weights[train_size + val_size:],
            'labels': self.y[train_size + val_size:]
        }
        
        return train_data, val_data, test_data


class PriceGraphDataLoader(Dataset):
    """PyTorch Dataset for price graph data."""
    
    def __init__(self, embeddings, weights, labels):
        """
        Initialize dataset.
        
        Args:
            embeddings: Graph embeddings
            weights: Node weights
            labels: Target labels
        """
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.weights[idx], self.labels[idx]


class InputAttention(nn.Module):
    """Input attention mechanism for selecting important input features."""
    
    def __init__(self, input_dim, hidden_dim):
        super(InputAttention, self).__init__()
        self.attention = nn.Linear(input_dim + hidden_dim*2, 1)
    
    def forward(self, h_prev, c_prev, x):
        """
        Forward pass.
        
        Args:
            h_prev: Previous hidden state
            c_prev: Previous cell state
            x: Input tensor of shape (batch_size, input_length, input_dim)
            
        Returns:
            Weighted input tensor
        """
        batch_size, seq_len, input_dim = x.size()
        
        # Repeat hidden state for each input feature
        h_expanded = h_prev.unsqueeze(1).repeat(1, seq_len, 1)
        c_expanded = c_prev.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Concatenate input with previous hidden and cell states
        combined = torch.cat([h_expanded, c_expanded, x], dim=2)
        
        # Calculate attention weights
        e = self.attention(combined).squeeze(2)
        alpha = F.softmax(e, dim=1).unsqueeze(2)
        
        # Apply attention weights
        weighted_input = alpha * x
        
        return weighted_input


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for selecting important timesteps."""
    
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Linear(hidden_dim*3, 1)
    
    def forward(self, h_prev, c_prev, encoder_states, node_weights=None):
        """
        Forward pass.
        
        Args:
            h_prev: Previous hidden state
            c_prev: Previous cell state
            encoder_states: All encoder hidden states
            node_weights: Optional node weights for enhancing attention
            
        Returns:
            Context vector
        """
        batch_size, seq_len, hidden_dim = encoder_states.size()
        
        # Expand decoder state for each timestep
        h_expanded = h_prev.unsqueeze(1).repeat(1, seq_len, 1)
        c_expanded = c_prev.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Concatenate with encoder states
        combined = torch.cat([h_expanded, c_expanded, encoder_states], dim=2)
        
        # Calculate attention scores
        e = self.attention(combined).squeeze(2)
        
        # If node weights are provided, enhance attention scores
        if node_weights is not None:
            # Normalize node weights
            norm_weights = F.softmax(node_weights, dim=1)
            # Add normalized weights to attention scores
            e = e + norm_weights
        
        # Apply softmax to get attention weights
        beta = F.softmax(e, dim=1).unsqueeze(2)
        
        # Calculate context vector
        context = torch.bmm(beta.transpose(1, 2), encoder_states).squeeze(1)
        
        return context


class DARNN(nn.Module):
    """Dual-Stage Attention-Based RNN with node weights enhancement."""
    
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(DARNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Input Attention
        self.input_attention = InputAttention(embedding_dim, hidden_dim)
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # Temporal Attention
        self.temporal_attention = TemporalAttention(hidden_dim)
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
    
    def forward(self, x, node_weights=None):
        """
        Forward pass with simplified processing.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)
            node_weights: Node weights tensor of shape (batch_size, seq_len)
            
        Returns:
            Final hidden state of the decoder
        """
        batch_size, seq_len, embedding_dim = x.size()
        
        # Initialize encoder states
        h_enc = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
        c_enc = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
        
        # Process entire sequence through encoder LSTM
        encoder_outputs, (h_enc, c_enc) = self.encoder_lstm(x, (h_enc, c_enc))
        
        # Initialize decoder states
        h_dec = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
        c_dec = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)
        
        # Apply temporal attention with node weights
        context = self.temporal_attention(
            h_dec.squeeze(0), c_dec.squeeze(0), 
            encoder_outputs, node_weights
        )
        
        # Decoder LSTM (one step)
        context = context.unsqueeze(1)  # Add time dimension
        _, (h_dec, _) = self.decoder_lstm(context, (h_dec, c_dec))
        
        return h_dec.squeeze(0)


class CrossAssetAttention(nn.Module):
    """Cross-Asset Attention Network for modeling stock interrelationships."""
    
    def __init__(self, hidden_dim):
        super(CrossAssetAttention, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Query, Key, Value projections
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Scaling factor
        self.scaling = torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, hidden_dim)
            
        Returns:
            Output tensor after self-attention
        """
        batch_size = x.size(0)
        
        # Project input to query, key, value
        q = self.query(x)  # (batch_size, hidden_dim)
        k = self.key(x)    # (batch_size, hidden_dim)
        v = self.value(x)  # (batch_size, hidden_dim)
        
        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(0, 1)) / self.scaling  # (batch_size, batch_size)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch_size, batch_size)
        
        # Calculate context vector
        context = torch.matmul(attn_weights, v)  # (batch_size, hidden_dim)
        
        return context


class PriceGraphModel(nn.Module):
    """Full Price Graph model combining multiple DARNNs and cross-asset attention."""
    
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_features=6):
        super(PriceGraphModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_features = num_features  # Number of price features (open, high, low, close, volume, amount)
        
        # Create a DARNN for each price feature
        self.darnns = nn.ModuleList([
            DARNN(input_dim, hidden_dim, embedding_dim) for _ in range(num_features)
        ])
        
        # Cross-Asset Attention Network
        self.caan = CrossAssetAttention(hidden_dim)
        
        # Final prediction layer
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x_embeddings, x_weights):
        """
        Forward pass with simplified processing.
        
        Args:
            x_embeddings: Graph embeddings tensor of shape (batch_size, num_features, seq_len, embedding_dim)
            x_weights: Node weights tensor of shape (batch_size, num_features, seq_len)
            
        Returns:
            Predicted probability of price increase
        """
        batch_size = x_embeddings.size(0)
        
        # Process each feature with its own DARNN
        feature_outputs = []
        for i in range(self.num_features):
            try:
                # Extract embeddings and weights for this feature
                embeddings = x_embeddings[:, i]  # (batch_size, seq_len, embedding_dim)
                weights = x_weights[:, i]        # (batch_size, seq_len)
                
                # Pass through DARNN
                output = self.darnns[i](embeddings, weights)
                feature_outputs.append(output)
            except Exception as e:
                # If there's an error with one feature, skip it
                logger.warning(f"Error processing feature {i}: {e}")
                continue
        
        if not feature_outputs:
            # If no features were processed successfully, return random predictions
            return torch.rand(batch_size).to(x_embeddings.device)
            
        # Aggregate feature outputs (element-wise addition as in the paper)
        combined_output = torch.stack(feature_outputs, dim=1).sum(dim=1)
        
        # Apply Cross-Asset Attention
        attended_output = self.caan(combined_output)
        
        # Final prediction
        logits = self.fc(attended_output)
        probabilities = torch.sigmoid(logits).squeeze(1)
        
        return probabilities


class BloombergDataLoader:
    """Load stock data from Bloomberg."""
    
    def __init__(self, tickers, start_date, end_date):
        """
        Initialize the data loader.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        
        # Check for Bloomberg API availability at initialization
        try:
            import blpapi
            self.blpapi = blpapi
            self.HAS_BLOOMBERG = True
            logger.info("Successfully imported Bloomberg API")
        except ImportError:
            self.blpapi = None
            self.HAS_BLOOMBERG = False
            logger.warning("Bloomberg API not available. Using synthetic data.")
    
    def load_data(self):
        """
        Load data from Bloomberg.
        
        Returns:
            Dictionary mapping tickers to DataFrames
        """
        if self.HAS_BLOOMBERG:
            try:
                return self._load_from_bloomberg()
            except Exception as e:
                logger.error(f"Error loading from Bloomberg: {e}")
                logger.warning("Falling back to synthetic data.")
                return self._generate_synthetic_data()
        else:
            return self._generate_synthetic_data()
    
    def _load_from_bloomberg(self):
        """
        Load data from Bloomberg API.
        
        Returns:
            Dictionary mapping tickers to DataFrames
        """
        logger.info("Loading data from Bloomberg...")
        
        if not self.HAS_BLOOMBERG or self.blpapi is None:
            raise ImportError("Bloomberg API not available")
        
        # Initialize Bloomberg API session
        session = self.blpapi.Session()
        if not session.start():
            logger.error("Failed to start Bloomberg session.")
            raise ConnectionError("Failed to start Bloomberg session")
        
        # Open Bloomberg API service
        if not session.openService("//blp/refdata"):
            logger.error("Failed to open //blp/refdata service.")
            session.stop()
            raise ConnectionError("Failed to open //blp/refdata service")
        
        # Get service
        refDataService = session.getService("//blp/refdata")
        
        # Process each ticker
        data = {}
        for ticker in self.tickers:
            # Create a new request for each ticker
            request = refDataService.createRequest("HistoricalDataRequest")
            
            # Set request parameters
            request.set("startDate", self.start_date.strftime("%Y%m%d"))
            request.set("endDate", self.end_date.strftime("%Y%m%d"))
            request.set("periodicitySelection", "DAILY")
            
            # Set fields
            fields = ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME", "EQY_TURNOVER"]
            for field in fields:
                request.append("fields", field)
            
            # Add the ticker
            request.append("securities", ticker)
            
            try:
                # Send request
                session.sendRequest(request)
                
                # Process response
                while True:
                    event = session.nextEvent(500)
                    if event.eventType() == self.blpapi.Event.RESPONSE:
                        for msg in event:
                            security_data = msg.getElement("securityData")
                            ticker_name = security_data.getElement("security").getValue()
                            field_data = security_data.getElement("fieldData")
                            
                            # Extract data
                            dates = []
                            opens = []
                            highs = []
                            lows = []
                            closes = []
                            volumes = []
                            amounts = []
                            
                            for i in range(field_data.numValues()):
                                field_value = field_data.getValue(i)
                                date = field_value.getElement("date").getValue()
                                dates.append(date.strftime("%Y-%m-%d"))
                                
                                opens.append(field_value.getElement("PX_OPEN").getValue())
                                highs.append(field_value.getElement("PX_HIGH").getValue())
                                lows.append(field_value.getElement("PX_LOW").getValue())
                                closes.append(field_value.getElement("PX_LAST").getValue())
                                volumes.append(field_value.getElement("PX_VOLUME").getValue())
                                amounts.append(field_value.getElement("EQY_TURNOVER").getValue())
                            
                            # Create DataFrame
                            df = pd.DataFrame({
                                'date': dates,
                                'open': opens,
                                'high': highs,
                                'low': lows,
                                'close': closes,
                                'volume': volumes,
                                'amount': amounts
                            })
                            
                            df['date'] = pd.to_datetime(df['date'])
                            df.set_index('date', inplace=True)
                            
                            data[ticker_name] = df
                        
                        break
                    
                    if event.eventType() == self.blpapi.Event.TIMEOUT:
                        break
            
            except Exception as e:
                logger.error(f"Error loading data for {ticker}: {e}")
        
        # Stop session
        session.stop()
        
        if not data:
            logger.error("No data retrieved from Bloomberg.")
            raise ValueError("No data retrieved from Bloomberg")
        
        return data
    
    def _generate_synthetic_data(self):
        """
        Generate synthetic stock data for testing.
        
        Returns:
            Dictionary mapping tickers to DataFrames
        """
        logger.info("Generating synthetic data...")
        
        # Create date range
        date_range = pd.date_range(self.start_date, self.end_date, freq='B')
        
        # Generate data for each ticker
        data = {}
        for ticker in self.tickers:
            # Set random seed based on ticker
            np.random.seed(hash(ticker) % 2**32)
            
            # Generate random walk for close price
            n = len(date_range)
            close = 100 + np.cumsum(np.random.normal(0.0005, 0.01, n))
            
            # Generate other prices based on close
            open_price = close * np.random.normal(1, 0.005, n)
            high = np.maximum(close, open_price) * np.random.normal(1.01, 0.005, n)
            low = np.minimum(close, open_price) * np.random.normal(0.99, 0.005, n)
            
            # Generate volume and amount
            volume = np.random.normal(1e6, 2e5, n)
            amount = volume * close * np.random.normal(1, 0.1, n)
            
            # Create DataFrame
            df = pd.DataFrame({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'amount': amount
            }, index=date_range)
            
            data[ticker] = df
        
        return data


def get_data_from_csv(tickers, csv_path):
    """Load stock data from a CSV file."""
    data_dict = {}
    try:
        # Load the CSV
        df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
        
        # If the CSV contains multiple tickers, split by ticker
        if 'ticker' in df.columns:
            for ticker in tickers:
                ticker_data = df[df['ticker'] == ticker].copy()
                if not ticker_data.empty:
                    # Drop the ticker column
                    ticker_data = ticker_data.drop(columns=['ticker'])
                    data_dict[ticker] = ticker_data
                    logger.info(f"Loaded {len(ticker_data)} rows for {ticker} from CSV")
                else:
                    logger.warning(f"No data found for {ticker} in CSV file")
        else:
            # Assume the CSV contains data for a single ticker
            # Use the first ticker in the list
            data_dict[tickers[0]] = df
            logger.info(f"Loaded {len(df)} rows for {tickers[0]} from CSV")
    except Exception as e:
        logger.error(f"Error loading data from CSV: {e}")
    
    return data_dict


def train_price_graph_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    """
    Train the Price Graph model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        lr: Learning rate
        
    Returns:
        Trained model and training history
    """
    # Move model to device
    model = model.to(device)
    
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # For early stopping
    best_val_loss = float('inf')
    patience = 5
    counter = 0
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    logger.info("Starting training...")
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for embeddings, weights, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)"):
            # Move data to device
            embeddings = embeddings.to(device)
            weights = weights.to(device)
            targets = targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(embeddings, weights)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track loss and predictions
            train_loss += loss.item()
            train_preds.extend((outputs > 0.5).float().cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for embeddings, weights, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Val)"):
                # Move data to device
                embeddings = embeddings.to(device)
                weights = weights.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(embeddings, weights)
                loss = criterion(outputs, targets)
                
                # Track loss and predictions
                val_loss += loss.item()
                val_preds.extend((outputs > 0.5).float().cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        
        # Print metrics
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_price_graph_model.pth')
        else:
            counter += 1
            if counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_price_graph_model.pth'))
    
    return model, history


def evaluate_model(model, test_loader):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for embeddings, weights, labels in tqdm(test_loader, desc="Evaluating"):
            # Move data to device
            embeddings = embeddings.to(device)
            weights = weights.to(device)
            
            # Get predictions
            outputs = model(embeddings, weights)
            predictions.extend((outputs > 0.5).float().cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions)
    recall = recall_score(targets, predictions)
    f1 = f1_score(targets, predictions)
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test Precision: {precision:.4f}")
    logger.info(f"Test Recall: {recall:.4f}")
    logger.info(f"Test F1 Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': predictions,
        'targets': targets
    }


def backtest_strategy(data, predictions, targets, initial_capital=10000):
    """
    Backtest trading strategy based on model predictions.
    
    Args:
        data: Price data
        predictions: Model predictions
        targets: True labels
        initial_capital: Initial capital
        
    Returns:
        DataFrame with backtest results
    """
    # Create a DataFrame for the trading simulation
    backtest_data = pd.DataFrame({
        'close': data['close'].iloc[-len(predictions):].values,
        'prediction': predictions,
        'actual': targets
    })
    
    # Calculate daily returns
    backtest_data['daily_return'] = backtest_data['close'].pct_change().fillna(0)
    
    # Strategy returns (long when prediction is 1, short when prediction is 0)
    backtest_data['strategy_return'] = backtest_data['daily_return'] * (2 * backtest_data['prediction'] - 1)
    
    # Calculate cumulative returns
    backtest_data['cum_market_return'] = (1 + backtest_data['daily_return']).cumprod() - 1
    backtest_data['cum_strategy_return'] = (1 + backtest_data['strategy_return']).cumprod() - 1
    
    # Calculate equity curves
    backtest_data['market_equity'] = initial_capital * (1 + backtest_data['cum_market_return'])
    backtest_data['strategy_equity'] = initial_capital * (1 + backtest_data['cum_strategy_return'])
    
    # Calculate drawdowns
    backtest_data['market_peak'] = backtest_data['market_equity'].cummax()
    backtest_data['strategy_peak'] = backtest_data['strategy_equity'].cummax()
    backtest_data['market_drawdown'] = (backtest_data['market_equity'] - backtest_data['market_peak']) / backtest_data['market_peak']
    backtest_data['strategy_drawdown'] = (backtest_data['strategy_equity'] - backtest_data['strategy_peak']) / backtest_data['strategy_peak']
    
    # Final equity
    final_market_equity = backtest_data['market_equity'].iloc[-1]
    final_strategy_equity = backtest_data['strategy_equity'].iloc[-1]
    
    # Total returns
    total_market_return = (final_market_equity / initial_capital) - 1
    total_strategy_return = (final_strategy_equity / initial_capital) - 1
    
    # Maximum drawdowns
    max_market_drawdown = backtest_data['market_drawdown'].min()
    max_strategy_drawdown = backtest_data['strategy_drawdown'].min()
    
    # Daily statistics
    avg_daily_market_return = backtest_data['daily_return'].mean()
    avg_daily_strategy_return = backtest_data['strategy_return'].mean()
    std_daily_market_return = backtest_data['daily_return'].std()
    std_daily_strategy_return = backtest_data['strategy_return'].std()
    
    # Sharpe ratios (assuming risk-free rate = 0)
    sharpe_market = avg_daily_market_return / std_daily_market_return * np.sqrt(252)
    sharpe_strategy = avg_daily_strategy_return / std_daily_strategy_return * np.sqrt(252)
    
    # Print summary
    logger.info(f"Backtest Results:")
    logger.info(f"Total Market Return: {total_market_return:.4f} ({total_market_return*100:.2f}%)")
    logger.info(f"Total Strategy Return: {total_strategy_return:.4f} ({total_strategy_return*100:.2f}%)")
    logger.info(f"Market Sharpe Ratio: {sharpe_market:.4f}")
    logger.info(f"Strategy Sharpe Ratio: {sharpe_strategy:.4f}")
    logger.info(f"Max Market Drawdown: {max_market_drawdown:.4f} ({max_market_drawdown*100:.2f}%)")
    logger.info(f"Max Strategy Drawdown: {max_strategy_drawdown:.4f} ({max_strategy_drawdown*100:.2f}%)")
    
    return backtest_data


def plot_backtest_results(backtest_data):
    """
    Plot backtest results.
    
    Args:
        backtest_data: DataFrame with backtest results
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot equity curves
    ax1.plot(backtest_data.index, backtest_data['market_equity'], label='Market')
    ax1.plot(backtest_data.index, backtest_data['strategy_equity'], label='Price Graph Strategy')
    ax1.set_title('Equity Curves')
    ax1.set_ylabel('Equity')
    ax1.legend()
    ax1.grid(True)
    
    # Plot drawdowns
    ax2.fill_between(backtest_data.index, 0, backtest_data['market_drawdown'], alpha=0.3, color='blue', label='Market Drawdown')
    ax2.fill_between(backtest_data.index, 0, backtest_data['strategy_drawdown'], alpha=0.3, color='orange', label='Strategy Drawdown')
    ax2.set_title('Drawdowns')
    ax2.set_ylabel('Drawdown')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('backtest_results.png')
    plt.show()


def run_price_graph_strategy(ticker, start_date, end_date, window_size=20, embedding_dim=32, hidden_dim=64, batch_size=32, epochs=50, data_source="bloomberg"):
    """
    Run the full Price Graph strategy for a ticker.
    
    Args:
        ticker: Ticker symbol
        start_date: Start date for data
        end_date: End date for data
        window_size: Lookback window size
        embedding_dim: Dimension of node embeddings
        hidden_dim: Hidden dimension of model
        batch_size: Training batch size
        epochs: Number of training epochs
        data_source: Source of data ("bloomberg", "csv", or "synthetic")
        
    Returns:
        Evaluation metrics and backtest results
    """
    # Load data from the specified source
    if data_source == "bloomberg":
        data_loader = BloombergDataLoader([ticker], start_date, end_date)
        data_dict = data_loader.load_data()
    elif data_source == "csv":
        csv_path = input("Enter path to CSV file: ")
        data_dict = get_data_from_csv([ticker], csv_path)
        if not data_dict:
            logger.warning("No data loaded from CSV. Falling back to synthetic data.")
            data_loader = BloombergDataLoader([ticker], start_date, end_date)
            data_dict = data_loader._generate_synthetic_data()
    else:  # "synthetic"
        data_loader = BloombergDataLoader([ticker], start_date, end_date)
        data_dict = data_loader._generate_synthetic_data()
    
    # Check if we have data for the ticker
    if ticker not in data_dict:
        logger.error(f"No data available for {ticker}. Exiting.")
        return None, None
    
    # Get data for the ticker
    data = data_dict[ticker]
    
    # Check if data is sufficient
    if len(data) < 50:  # Arbitrary minimum length
        logger.error(f"Insufficient data for {ticker} (only {len(data)} rows). Exiting.")
        return None, None
    
    logger.info(f"Using {len(data)} days of data for {ticker} from {data.index.min()} to {data.index.max()}")
    
    # Create price graph dataset
    dataset = PriceGraphDataset(data, window_size=window_size, embedding_dim=embedding_dim)
    
    # Check if dataset was created successfully
    if len(dataset.y) == 0:
        logger.error("Failed to create dataset. No samples generated.")
        return None, None
    
    # Split data
    train_data, val_data, test_data = dataset.split_data()
    
    if train_data is None:
        logger.error("Failed to split data. Dataset may be empty.")
        return None, None
    
    # Create data loaders
    train_loader = DataLoader(
        PriceGraphDataLoader(train_data['embeddings'], train_data['weights'], train_data['labels']),
        batch_size=batch_size, shuffle=True
    )
    
    val_loader = DataLoader(
        PriceGraphDataLoader(val_data['embeddings'], val_data['weights'], val_data['labels']),
        batch_size=batch_size
    )
    
    test_loader = DataLoader(
        PriceGraphDataLoader(test_data['embeddings'], test_data['weights'], test_data['labels']),
        batch_size=batch_size
    )
    
    # Create model
    model = PriceGraphModel(window_size, hidden_dim, embedding_dim)
    
    # Train model
    model, history = train_price_graph_model(model, train_loader, val_loader, epochs=epochs)
    
    # Evaluate model
    eval_metrics = evaluate_model(model, test_loader)
    
    # Backtest strategy
    backtest_results = backtest_strategy(data, eval_metrics['predictions'], eval_metrics['targets'])
    
    # Plot backtest results
    plot_backtest_results(backtest_results)
    
    return eval_metrics, backtest_results


def main():
    """Main function."""
    # Define ticker and date range
    ticker = "AAPL US Equity"  # Example ticker
    start_date = datetime(2018, 1, 1)
    end_date = datetime(2020, 12, 31)
    
    # Ask user for data source
    print("\nChoose data source:")
    print("1. Bloomberg (requires Bloomberg API)")
    print("2. CSV file")
    print("3. Synthetic data (for testing)")
    
    choice = input("Enter choice (1-3): ")
    
    if choice == "1":
        data_source = "bloomberg"
    elif choice == "2":
        data_source = "csv"
    else:
        data_source = "synthetic"
    
    # Run strategy
    eval_metrics, backtest_results = run_price_graph_strategy(
        ticker, start_date, end_date, 
        window_size=20, embedding_dim=32, hidden_dim=64, 
        batch_size=32, epochs=50, data_source=data_source
    )
    
    if eval_metrics is None:
        logger.error("Strategy execution failed.")
        return
    
    # Print final results
    logger.info("Price Graph Strategy Results:")
    logger.info(f"Accuracy: {eval_metrics['accuracy']:.4f}")
    logger.info(f"Precision: {eval_metrics['precision']:.4f}")
    logger.info(f"Recall: {eval_metrics['recall']:.4f}")
    logger.info(f"F1 Score: {eval_metrics['f1']:.4f}")
    
    # Calculate final returns
    final_market_return = (backtest_results['market_equity'].iloc[-1] / backtest_results['market_equity'].iloc[0]) - 1
    final_strategy_return = (backtest_results['strategy_equity'].iloc[-1] / backtest_results['strategy_equity'].iloc[0]) - 1
    
    logger.info(f"Market Return: {final_market_return:.4f} ({final_market_return*100:.2f}%)")
    logger.info(f"Strategy Return: {final_strategy_return:.4f} ({final_strategy_return*100:.2f}%)")


if __name__ == "__main__":
    main()