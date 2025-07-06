import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import networkx as nx
import random
from datetime import datetime, timedelta

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Parameters for simulation
n_companies = 100  # Total number of companies
n_frequent_acquirers = 20  # Number of frequent acquirers
n_years = 5  # Simulation period in years
n_features = 10  # Number of financial features
n_text_features = 50  # Dimension of text embeddings
n_ma_events = 200  # Total number of M&A events to simulate
similarity_threshold = 0.3  # Threshold for TNIC similarity

# Generate dates for simulation
start_date = datetime(2016, 1, 1)
end_date = datetime(2020, 12, 31)
simulation_days = (end_date - start_date).days + 1

print("Generating simulated data...")

# Generate company data
companies = {
    i: {
        'id': i,
        'name': f'Company_{i}',
        'is_frequent_acquirer': i < n_frequent_acquirers,
        'size': np.random.uniform(1, 100) * (2 if i < n_frequent_acquirers else 1),  # Acquirers tend to be larger
        'age': np.random.randint(1, 30) * (1.5 if i < n_frequent_acquirers else 1)   # Acquirers tend to be older
    }
    for i in range(n_companies)
}

# Function to generate financial features for a company at a given time
def generate_financial_features(company_id, time_idx, prev_features=None):
    if prev_features is None:
        # Initial features
        base = np.random.normal(0, 1, n_features)
        
        # Acquirers tend to have better financials
        if company_id < n_frequent_acquirers:
            base += np.random.uniform(0.2, 0.5, n_features)
    else:
        # Evolution of features with some random drift
        base = prev_features + np.random.normal(0, 0.1, n_features)
    
    return base

# Function to generate text embeddings (simulating 10-K "Business Description")
def generate_text_embeddings(company_id, time_idx, prev_embeddings=None):
    if prev_embeddings is None:
        # Initial embeddings
        base = np.random.normal(0, 1, n_text_features)
        
        # Industry clusters - companies in same industry have similar descriptions
        industry_factor = company_id % 10  # 10 industries
        industry_bias = np.random.normal(industry_factor, 0.5, n_text_features)
        base += industry_bias
    else:
        # Text embeddings change slowly over time
        base = prev_embeddings + np.random.normal(0, 0.05, n_text_features)
    
    return base / np.linalg.norm(base)  # Normalize

# Generate TNIC similarity matrix (industry network)
def generate_tnic_matrix(time_idx):
    tnic_matrix = np.zeros((n_companies, n_companies))
    
    # Base similarity based on industry (companies in same industry are more similar)
    for i in range(n_companies):
        for j in range(n_companies):
            if i == j:
                tnic_matrix[i, j] = 1.0
            else:
                # Companies in same industry have higher similarity
                industry_similarity = 0.7 if (i % 10) == (j % 10) else 0.1
                # Add some random variation
                tnic_matrix[i, j] = max(0, min(1, industry_similarity + np.random.normal(0, 0.1)))
    
    # Evolve TNIC similarity over time (slight random changes)
    if time_idx > 0:
        tnic_matrix += np.random.normal(0, 0.02, (n_companies, n_companies))
        np.fill_diagonal(tnic_matrix, 1.0)  # Self-similarity is always 1
        tnic_matrix = np.clip(tnic_matrix, 0, 1)  # Keep values between 0 and 1
    
    return tnic_matrix

# Generate time series data for all companies
financial_features = {}
text_embeddings = {}
tnic_matrices = {}

# Quarterly data points
time_points = pd.date_range(start=start_date, end=end_date, freq='Q')
for t_idx, t in enumerate(time_points):
    financial_features[t] = {}
    text_embeddings[t] = {}
    
    for company_id in range(n_companies):
        if t_idx == 0:
            financial_features[t][company_id] = generate_financial_features(company_id, t_idx)
            text_embeddings[t][company_id] = generate_text_embeddings(company_id, t_idx)
        else:
            prev_t = time_points[t_idx-1]
            financial_features[t][company_id] = generate_financial_features(
                company_id, t_idx, financial_features[prev_t][company_id])
            text_embeddings[t][company_id] = generate_text_embeddings(
                company_id, t_idx, text_embeddings[prev_t][company_id])
    
    # Generate TNIC matrix for this time point
    tnic_matrices[t] = generate_tnic_matrix(t_idx)

# Simulate M&A events
def simulate_ma_events(n_events):
    events = []
    
    # Each frequent acquirer has a base acquisition rate
    acquisition_rates = [0.01 + np.random.uniform(0, 0.03) for _ in range(n_frequent_acquirers)]
    
    # Peer effects: recent M&A activities increase probability of further M&As
    peer_effect_decay = 0.8  # How quickly peer effect decays over time
    
    # Industry consolidation waves (certain industries have higher M&A activity in periods)
    industry_waves = {i: np.random.normal(0, 1) for i in range(10)}
    
    # Keep track of last M&A for each acquirer
    last_ma_time = {i: start_date for i in range(n_frequent_acquirers)}
    
    # Current time in simulation
    current_time = start_date
    
    while len(events) < n_events and current_time < end_date:
        # Advance time by one day
        current_time += timedelta(days=1)
        
        # Find closest data point for company features
        closest_data_point = max([t for t in time_points if t <= current_time])
        
        # Check each frequent acquirer for potential M&A activity
        for acquirer_id in range(n_frequent_acquirers):
            # Base rate adjusted by time since last acquisition
            days_since_last_ma = (current_time - last_ma_time[acquirer_id]).days
            time_factor = min(1.0, days_since_last_ma / 180)  # Increases up to 6 months
            
            # Financial condition factor (better financials increase M&A probability)
            financials = financial_features[closest_data_point][acquirer_id]
            financial_factor = 1.0 + 0.2 * (np.mean(financials) / 2)
            
            # Industry wave factor
            acquirer_industry = acquirer_id % 10
            wave_phase = np.sin((current_time - start_date).days / 180 + industry_waves[acquirer_industry])
            industry_factor = 1.0 + 0.3 * (wave_phase + 1) / 2  # Normalize to [1.0, 1.3]
            
            # Peer effect from recent M&As in same industry
            peer_effect = 1.0
            for event in reversed(events):
                if event['acquirer_id'] % 10 == acquirer_industry:  # Same industry
                    days_ago = (current_time - event['date']).days
                    if days_ago < 90:  # Peer effect lasts 3 months
                        peer_effect += 0.5 * (peer_effect_decay ** (days_ago / 30))
            
            # Combined probability of acquisition on this day
            daily_prob = min(0.8, acquisition_rates[acquirer_id] * time_factor * financial_factor * 
                            industry_factor * peer_effect)
            
            if random.random() < daily_prob:
                # Decide on target
                tnic_matrix = tnic_matrices[closest_data_point]
                
                # Exclude frequent acquirers and already acquired companies from targets
                acquired_companies = set([e['target_id'] for e in events])
                potential_targets = [i for i in range(n_companies) 
                                   if i >= n_frequent_acquirers and i not in acquired_companies]
                
                if not potential_targets:
                    continue
                
                # Calculate target probabilities based on TNIC similarity and other factors
                target_probs = []
                for target_id in potential_targets:
                    # TNIC similarity factor
                    similarity = tnic_matrix[acquirer_id, target_id]
                    similarity_factor = similarity * 2  # Higher similarity increases probability
                    
                    # Size complementarity (prefer smaller targets)
                    target_size = companies[target_id]['size']
                    acquirer_size = companies[acquirer_id]['size']
                    size_ratio = target_size / acquirer_size
                    size_factor = max(0.1, 1.0 - size_ratio)  # Prefer smaller targets
                    
                    # Combined probability
                    target_prob = similarity_factor * size_factor
                    target_probs.append(target_prob)
                
                # Normalize probabilities
                target_probs = np.array(target_probs)
                if np.sum(target_probs) > 0:
                    target_probs = target_probs / np.sum(target_probs)
                    
                    # Select target
                    target_idx = np.random.choice(len(potential_targets), p=target_probs)
                    target_id = potential_targets[target_idx]
                    
                    # Record the M&A event
                    events.append({
                        'date': current_time,
                        'acquirer_id': acquirer_id,
                        'target_id': target_id,
                        'tnic_similarity': tnic_matrix[acquirer_id, target_id]
                    })
                    
                    # Update last M&A time for acquirer
                    last_ma_time[acquirer_id] = current_time
    
    return events

# Generate M&A events
ma_events = simulate_ma_events(n_ma_events)
print(f"Generated {len(ma_events)} M&A events")

# Sort events by date
ma_events.sort(key=lambda x: x['date'])

# Convert events to DataFrame for easier analysis
events_df = pd.DataFrame(ma_events)
print(events_df.head())

# Plot M&A events timeline
plt.figure(figsize=(12, 6))
for i in range(n_frequent_acquirers):
    acquirer_events = events_df[events_df['acquirer_id'] == i]
    if not acquirer_events.empty:
        plt.scatter(acquirer_events['date'], [i] * len(acquirer_events), label=f'Acquirer {i}' if i < 5 else None)

plt.yticks(range(n_frequent_acquirers))
plt.xlabel('Date')
plt.ylabel('Acquirer ID')
plt.title('M&A Events Timeline')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('ma_events_timeline.png')

# Plot industry network (using last TNIC matrix)
last_time_point = time_points[-1]
last_tnic = tnic_matrices[last_time_point]

# Create a graph showing the strongest connections
G = nx.Graph()
for i in range(n_companies):
    G.add_node(i, size=companies[i]['size'])

# Add edges for pairs with similarity above threshold
for i in range(n_companies):
    for j in range(i+1, n_companies):
        if last_tnic[i, j] > similarity_threshold:
            G.add_edge(i, j, weight=last_tnic[i, j])

# Plot the network
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, seed=42)
node_sizes = [companies[i]['size'] * 10 for i in range(n_companies)]
node_colors = ['red' if i < n_frequent_acquirers else 'blue' for i in range(n_companies)]

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.7)
edges = nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)
plt.axis('off')
plt.title('Industry Network (TNIC Similarity)')
plt.tight_layout()
plt.savefig('industry_network.png')

# Now, let's implement the TDIN model from the paper

# Neural network components for the model
class IntrinsicFactorsEncoder(nn.Module):
    def __init__(self, n_financial_features, n_text_features, hidden_dim, output_dim):
        super().__init__()
        self.financial_encoder = nn.Sequential(
            nn.Linear(n_financial_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim // 2)
        )
        
        self.text_encoder = nn.Sequential(
            nn.Linear(n_text_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim // 2)
        )
        
    def forward(self, financial_features, text_features):
        financial_embedding = self.financial_encoder(financial_features)
        text_embedding = self.text_encoder(text_features)
        return torch.cat([financial_embedding, text_embedding], dim=1)

class ExtrinsicFactorsEncoder(nn.Module):
    def __init__(self, event_dim, hidden_dim, output_dim):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=event_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, event_sequences, seq_lengths):
        # Pack padded sequence for efficient RNN computation
        packed_sequences = nn.utils.rnn.pack_padded_sequence(
            event_sequences, seq_lengths, batch_first=True, enforce_sorted=False
        )
        
        _, hidden = self.rnn(packed_sequences)
        return self.fc(hidden.squeeze(0))

class GraphMessagePassing(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, node_features, adj_matrix):
        # First message passing layer
        aggregated = torch.matmul(adj_matrix, node_features)
        h1 = F.relu(self.fc1(aggregated))
        
        # Second message passing layer
        aggregated2 = torch.matmul(adj_matrix, h1)
        h2 = self.fc2(aggregated2)
        
        return h2

class TimingModule(nn.Module):
    def __init__(self, intrinsic_dim, extrinsic_dim):
        super().__init__()
        self.we = nn.Parameter(torch.randn(intrinsic_dim))
        self.wc = nn.Parameter(torch.randn(extrinsic_dim))
        self.wo1 = nn.Parameter(torch.ones(1))
        self.wo2 = nn.Parameter(torch.ones(1))
        
    def forward(self, intrinsic_factors, extrinsic_factors, time_delta):
        base_rate = torch.matmul(intrinsic_factors, self.we) + torch.matmul(extrinsic_factors, self.wc)
        time_factor = self.wo1 * torch.exp(-self.wo2 * time_delta)
        intensity = F.softplus(base_rate + time_factor)  # Ensure positive intensity
        return intensity

class ChoiceModule(nn.Module):
    def __init__(self, node_dim, hidden_dim):
        super().__init__()
        self.message_passing = GraphMessagePassing(node_dim, hidden_dim, node_dim)
        self.compatibility_matrix = nn.Parameter(torch.randn(node_dim, node_dim))
        
    def forward(self, node_features, adj_matrix):
        # Apply message passing to get updated node embeddings
        node_embeddings = self.message_passing(node_features, adj_matrix)
        
        # Compute compatibility scores between acquirers and targets
        n_nodes = node_embeddings.shape[0]
        scores = torch.zeros((n_nodes, n_nodes), device=node_embeddings.device)
        
        for i in range(n_nodes):
            acquirer_embedding = node_embeddings[i].unsqueeze(0)  # [1, dim]
            compatibility = torch.matmul(
                torch.matmul(acquirer_embedding, self.compatibility_matrix),
                node_embeddings.t()
            )
            scores[i] = compatibility.squeeze(0)
        
        return scores

class TDIN(nn.Module):
    def __init__(self, n_financial_features, n_text_features, event_dim, hidden_dim, embedding_dim):
        super().__init__()
        
        # Intrinsic factors encoder
        self.intrinsic_encoder = IntrinsicFactorsEncoder(
            n_financial_features, n_text_features, hidden_dim, embedding_dim
        )
        
        # Extrinsic factors encoder
        self.extrinsic_encoder = ExtrinsicFactorsEncoder(
            event_dim, hidden_dim, embedding_dim
        )
        
        # Timing module
        self.timing_module = TimingModule(embedding_dim, embedding_dim)
        
        # Choice module
        self.choice_module = ChoiceModule(embedding_dim, hidden_dim)
        
    def forward(self, financial_features, text_features, event_sequences, seq_lengths, 
                time_delta, adj_matrix, available_mask=None):
        """
        Parameters:
        - financial_features: Financial data for each company [batch_size, n_financial_features]
        - text_features: Text embeddings for each company [batch_size, n_text_features]
        - event_sequences: Sequences of M&A events [batch_size, seq_len, event_dim]
        - seq_lengths: Length of each event sequence [batch_size]
        - time_delta: Time since last event for each company [batch_size]
        - adj_matrix: TNIC similarity matrix [n_companies, n_companies]
        - available_mask: Mask for available target companies [n_companies]
        
        Returns:
        - intensity: M&A intensity for each acquirer [batch_size]
        - choice_probs: Target selection probabilities [batch_size, n_companies]
        """
        # Encode intrinsic factors
        intrinsic_embeddings = self.intrinsic_encoder(financial_features, text_features)
        
        # Encode extrinsic factors
        extrinsic_embeddings = self.extrinsic_encoder(event_sequences, seq_lengths)
        
        # Compute M&A intensity (timing module)
        intensity = self.timing_module(intrinsic_embeddings, extrinsic_embeddings, time_delta)
        
        # Apply choice module to compute target probabilities
        choice_scores = self.choice_module(intrinsic_embeddings, adj_matrix)
        
        # Apply available mask if provided
        if available_mask is not None:
            choice_scores = choice_scores * available_mask
            
        # Apply softmax to get probabilities
        choice_probs = F.softmax(choice_scores, dim=1)
        
        return intensity, choice_probs

# Prepare data for model training
class MADataset(Dataset):
    def __init__(self, ma_events, financial_features, text_embeddings, tnic_matrices, companies,
                 n_companies, n_frequent_acquirers, max_seq_length=10):
        self.ma_events = ma_events
        self.financial_features = financial_features
        self.text_embeddings = text_embeddings
        self.tnic_matrices = tnic_matrices
        self.companies = companies
        self.n_companies = n_companies
        self.n_frequent_acquirers = n_frequent_acquirers
        self.max_seq_length = max_seq_length
        
        # Group events by acquirer
        self.acquirer_events = {i: [] for i in range(n_frequent_acquirers)}
        for event in ma_events:
            self.acquirer_events[event['acquirer_id']].append(event)
        
        # For each acquirer, create training examples
        self.examples = []
        for acquirer_id in range(n_frequent_acquirers):
            events = self.acquirer_events[acquirer_id]
            if len(events) < 2:  # Need at least two events for prediction
                continue
                
            for i in range(1, len(events)):
                prev_event = events[i-1]
                curr_event = events[i]
                
                # Find closest data point before current event
                time_points = list(financial_features.keys())
                data_point = max([t for t in time_points if t <= curr_event['date']])
                
                # Get available targets at this time
                acquired_companies = set([e['target_id'] for e in ma_events if e['date'] < curr_event['date']])
                available_targets = [i for i in range(n_companies) 
                                   if i >= n_frequent_acquirers and i not in acquired_companies]
                
                # Create example
                example = {
                    'acquirer_id': acquirer_id,
                    'target_id': curr_event['target_id'],
                    'date': curr_event['date'],
                    'prev_date': prev_event['date'],
                    'data_point': data_point,
                    'available_targets': available_targets,
                    'time_delta': (curr_event['date'] - prev_event['date']).days / 365.0  # Convert to years
                }
                
                # Collect previous events for sequence
                prev_events = [e for e in ma_events if e['date'] < curr_event['date'] and 
                              (e['acquirer_id'] == acquirer_id or 
                               self.companies[e['acquirer_id']]['is_frequent_acquirer'])]
                prev_events.sort(key=lambda x: x['date'])
                example['prev_events'] = prev_events[-self.max_seq_length:] if prev_events else []
                
                self.examples.append(example)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        acquirer_id = example['acquirer_id']
        target_id = example['target_id']
        data_point = example['data_point']
        
        # Prepare financial and text features
        financial_features = torch.tensor(
            [self.financial_features[data_point][i] for i in range(self.n_companies)],
            dtype=torch.float32
        )
        
        text_features = torch.tensor(
            [self.text_embeddings[data_point][i] for i in range(self.n_companies)],
            dtype=torch.float32
        )
        
        # Prepare event sequence
        prev_events = example['prev_events']
        seq_length = len(prev_events)
        
        event_sequence = torch.zeros((self.max_seq_length, 3), dtype=torch.float32)
        for i, event in enumerate(prev_events):
            if i >= self.max_seq_length:
                break
                
            # Event type: 0 for self-triggered, 1 for peer effect
            event_type = 0 if event['acquirer_id'] == acquirer_id else 1
            
            # Time since last event
            if i > 0:
                time_diff = (event['date'] - prev_events[i-1]['date']).days / 365.0  # Years
            else:
                time_diff = 0
                
            # TNIC similarity
            similarity = event['tnic_similarity']
            
            event_sequence[i] = torch.tensor([time_diff, event_type, similarity])
        
        # Prepare TNIC matrix
        tnic_matrix = torch.tensor(self.tnic_matrices[data_point], dtype=torch.float32)
        
        # Prepare available targets mask
        available_mask = torch.zeros(self.n_companies, dtype=torch.float32)
        for target in example['available_targets']:
            available_mask[target] = 1.0
        
        # Target label
        target_label = torch.zeros(self.n_companies, dtype=torch.float32)
        target_label[target_id] = 1.0
        
        return {
            'acquirer_id': acquirer_id,
            'financial_features': financial_features,
            'text_features': text_features,
            'event_sequence': event_sequence,
            'seq_length': seq_length,
            'time_delta': torch.tensor(example['time_delta'], dtype=torch.float32),
            'tnic_matrix': tnic_matrix,
            'available_mask': available_mask,
            'target_label': target_label
        }

# Create dataset
dataset = MADataset(
    ma_events, financial_features, text_embeddings, tnic_matrices, companies,
    n_companies, n_frequent_acquirers
)

# Split into train and test sets (80/20 split)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
)

print(f"Training examples: {len(train_dataset)}, Test examples: {len(test_dataset)}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize model
model = TDIN(
    n_financial_features=n_features,
    n_text_features=n_text_features,
    event_dim=3,  # time_diff, event_type, similarity
    hidden_dim=64,
    embedding_dim=32
)

# Training parameters
optimizer = optim.Adam(model.parameters(), lr=0.001)
n_epochs = 10

# Loss functions
def timing_loss(predicted_intensity, time_delta):
    # Negative log-likelihood for point process
    log_intensity = torch.log(predicted_intensity + 1e-10)
    integral_term = predicted_intensity * time_delta  # Simplified integration
    return -torch.mean(log_intensity - integral_term)

def choice_loss(predicted_probs, target_labels):
    # Cross-entropy loss for target selection
    return F.binary_cross_entropy(predicted_probs, target_labels)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(f"Training model on {device}...")

train_losses = []
test_losses = []

for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
        # Move batch data to device
        financial_features = batch['financial_features'].to(device)
        text_features = batch['text_features'].to(device)
        event_sequence = batch['event_sequence'].to(device)
        seq_length = batch['seq_length'].to(device)
        time_delta = batch['time_delta'].to(device)
        tnic_matrix = batch['tnic_matrix'].to(device)
        available_mask = batch['available_mask'].to(device)
        target_label = batch['target_label'].to(device)
        
        # Forward pass
        intensity, choice_probs = model(
            financial_features, text_features, event_sequence, seq_length,
            time_delta, tnic_matrix, available_mask
        )
        
        # Calculate losses
        t_loss = timing_loss(intensity, time_delta)
        c_loss = choice_loss(choice_probs, target_label)
        loss = t_loss + c_loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Evaluate on test set
    model.eval()
    test_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch data to device
            financial_features = batch['financial_features'].to(device)
            text_features = batch['text_features'].to(device)
            event_sequence = batch['event_sequence'].to(device)
            seq_length = batch['seq_length'].to(device)
            time_delta = batch['time_delta'].to(device)
            tnic_matrix = batch['tnic_matrix'].to(device)
            available_mask = batch['available_mask'].to(device)
            target_label = batch['target_label'].to(device)
            
            # Forward pass
            intensity, choice_probs = model(
                financial_features, text_features, event_sequence, seq_length,
                time_delta, tnic_matrix, available_mask
            )
            
            # Calculate losses
            t_loss = timing_loss(intensity, time_delta)
            c_loss = choice_loss(choice_probs, target_label)
            loss = t_loss + c_loss
            
            test_loss += loss.item()
            
            # Collect predictions and targets for AUC calculation
            for i in range(len(batch['acquirer_id'])):
                acquirer_id = batch['acquirer_id'][i].item()
                probs = choice_probs[i].cpu().numpy()
                labels = target_label[i].cpu().numpy()
                
                # Only consider available targets
                mask = available_mask[i].cpu().numpy() > 0
                if np.sum(mask) > 0:  # If there are available targets
                    all_predictions.append(probs[mask])
                    all_targets.append(labels[mask])
    
    avg_test_loss = test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    
    # Calculate AUC for target selection
    flat_preds = np.concatenate(all_predictions)
    flat_targets = np.concatenate(all_targets)
    auc_score = roc_auc_score(flat_targets, flat_preds)
    
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}, AUC = {auc_score:.4f}")

# Plot training and test loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_epochs+1), train_losses, 'b-', label='Training Loss')
plt.plot(range(1, n_epochs+1), test_losses, 'r-', label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('training_loss.png')

# Final evaluation
model.eval()
all_predictions = []
all_targets = []
all_intensities = []
all_time_deltas = []

with torch.no_grad():
    for batch in test_loader:
        # Move batch data to device
        financial_features = batch['financial_features'].to(device)
        text_features = batch['text_features'].to(device)
        event_sequence = batch['event_sequence'].to(device)
        seq_length = batch['seq_length'].to(device)
        time_delta = batch['time_delta'].to(device)
        tnic_matrix = batch['tnic_matrix'].to(device)
        available_mask = batch['available_mask'].to(device)
        target_label = batch['target_label'].to(device)
        
        # Forward pass
        intensity, choice_probs = model(
            financial_features, text_features, event_sequence, seq_length,
            time_delta, tnic_matrix, available_mask
        )
        
        # Collect predictions and targets
        for i in range(len(batch['acquirer_id'])):
            acquirer_id = batch['acquirer_id'][i].item()
            probs = choice_probs[i].cpu().numpy()
            labels = target_label[i].cpu().numpy()
            
            # Only consider available targets
            mask = available_mask[i].cpu().numpy() > 0
            if np.sum(mask) > 0:
                all_predictions.append(probs[mask])
                all_targets.append(labels[mask])
            
            all_intensities.append(intensity[i].item())
            all_time_deltas.append(time_delta[i].item())

# Calculate final AUC
flat_preds = np.concatenate(all_predictions)
flat_targets = np.concatenate(all_targets)
final_auc = roc_auc_score(flat_targets, flat_preds)
print(f"Final AUC Score: {final_auc:.4f}")

# Compare with baseline (acquisition likelihood model)
# Implement a simple logistic regression as baseline
from sklearn.linear_model import LogisticRegression

# Prepare data for logistic regression
X_train, y_train = [], []
X_test, y_test = [], []

# Extract features for baseline model
for i, example in enumerate(dataset.examples):
    acquirer_id = example['acquirer_id']
    target_id = example['target_id']
    data_point = example['data_point']
    
    # Get financial features for acquirer
    acquirer_financials = financial_features[data_point][acquirer_id]
    
    # Add industry features (using average TNIC similarity)
    tnic_matrix = tnic_matrices[data_point]
    acquirer_industry = acquirer_id % 10  # Simplified industry assignment
    
    # Add lag terms for industry M&A activity
    # Count M&A events in same industry in last 90, 180, 365 days
    industry_ma_90d = 0
    industry_ma_180d = 0
    industry_ma_365d = 0
    
    for event in ma_events:
        if event['date'] >= example['date']:
            continue
            
        event_acquirer_industry = event['acquirer_id'] % 10
        days_diff = (example['date'] - event['date']).days
        
        if event_acquirer_industry == acquirer_industry:
            if days_diff <= 90:
                industry_ma_90d += 1
            if days_diff <= 180:
                industry_ma_180d += 1
            if days_diff <= 365:
                industry_ma_365d += 1
    
    # Create feature vector for baseline model
    features = list(acquirer_financials) + [
        companies[acquirer_id]['size'],
        companies[acquirer_id]['age'],
        np.mean(tnic_matrix[acquirer_id]),
        industry_ma_90d,
        industry_ma_180d,
        industry_ma_365d
    ]
    
    # Add to appropriate dataset
    if i in train_dataset.indices:
        X_train.append(features)
        y_train.append(1 if target_id in example['available_targets'] else 0)
    else:
        X_test.append(features)
        y_test.append(1 if target_id in example['available_targets'] else 0)

# Train baseline model
baseline_model = LogisticRegression(max_iter=1000)
baseline_model.fit(X_train, y_train)

# Evaluate baseline model
baseline_probs = baseline_model.predict_proba(X_test)[:, 1]
baseline_auc = roc_auc_score(y_test, baseline_probs)

print(f"Baseline (Acquisition Likelihood) AUC: {baseline_auc:.4f}")
print(f"TDIN Model AUC: {final_auc:.4f}")
print(f"Performance Improvement: {(final_auc - baseline_auc) / baseline_auc * 100:.2f}%")

# Plot ROC curves for comparison
from sklearn.metrics import roc_curve

# Calculate ROC curve for TDIN model
fpr_tdin, tpr_tdin, _ = roc_curve(flat_targets, flat_preds)

# Calculate ROC curve for baseline model
fpr_baseline, tpr_baseline, _ = roc_curve(y_test, baseline_probs)

plt.figure(figsize=(10, 8))
plt.plot(fpr_tdin, tpr_tdin, 'b-', label=f'TDIN Model (AUC = {final_auc:.4f})')
plt.plot(fpr_baseline, tpr_baseline, 'r-', label=f'Acquisition Likelihood (AUC = {baseline_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('roc_comparison.png')

# Perform ablation study to understand component contributions
def run_ablation_test(ablation_name, ablation_model):
    # Evaluate on test set
    ablation_model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch data to device
            financial_features = batch['financial_features'].to(device)
            text_features = batch['text_features'].to(device)
            event_sequence = batch['event_sequence'].to(device)
            seq_length = batch['seq_length'].to(device)
            time_delta = batch['time_delta'].to(device)
            tnic_matrix = batch['tnic_matrix'].to(device)
            available_mask = batch['available_mask'].to(device)
            target_label = batch['target_label'].to(device)
            
            # Forward pass
            intensity, choice_probs = ablation_model(
                financial_features, text_features, event_sequence, seq_length,
                time_delta, tnic_matrix, available_mask
            )
            
            # Collect predictions and targets
            for i in range(len(batch['acquirer_id'])):
                mask = available_mask[i].cpu().numpy() > 0
                if np.sum(mask) > 0:
                    all_preds.append(choice_probs[i][mask].cpu().numpy())
                    all_labels.append(target_label[i][mask].cpu().numpy())
    
    # Calculate AUC
    flat_preds = np.concatenate(all_preds)
    flat_labels = np.concatenate(all_labels)
    auc = roc_auc_score(flat_labels, flat_preds)
    
    return auc

# Create ablation models
# 1. Without textual embeddings
class NoTextModel(TDIN):
    def forward(self, financial_features, text_features, event_sequences, seq_lengths, 
                time_delta, adj_matrix, available_mask=None):
        # Replace text features with zeros
        text_features = torch.zeros_like(text_features)
        return super().forward(financial_features, text_features, event_sequences, seq_lengths,
                              time_delta, adj_matrix, available_mask)

# 2. Without dynamic industry network
class NoDynamicNetworkModel(TDIN):
    def forward(self, financial_features, text_features, event_sequences, seq_lengths, 
                time_delta, adj_matrix, available_mask=None):
        # Replace network with identity matrix
        adj_matrix = torch.eye(adj_matrix.shape[0], device=adj_matrix.device)
        return super().forward(financial_features, text_features, event_sequences, seq_lengths,
                              time_delta, adj_matrix, available_mask)

# 3. Without peer effect modeling
class NoPeerEffectModel(TDIN):
    def forward(self, financial_features, text_features, event_sequences, seq_lengths, 
                time_delta, adj_matrix, available_mask=None):
        # Set all events to self-triggered (type=0)
        for i in range(event_sequences.shape[0]):
            for j in range(event_sequences.shape[1]):
                if event_sequences[i, j, 1] > 0:  # If event type is peer effect
                    event_sequences[i, j, 1] = 0  # Set to self-triggered
        
        return super().forward(financial_features, text_features, event_sequences, seq_lengths,
                              time_delta, adj_matrix, available_mask)

# 4. Without RNN in timing module
class NoRNNModel(TDIN):
    def __init__(self, n_financial_features, n_text_features, event_dim, hidden_dim, embedding_dim):
        super().__init__(n_financial_features, n_text_features, event_dim, hidden_dim, embedding_dim)
        # Replace RNN with simple averaging
        self.extrinsic_encoder = nn.Sequential(
            nn.Linear(event_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, financial_features, text_features, event_sequences, seq_lengths, 
                time_delta, adj_matrix, available_mask=None):
        # Encode intrinsic factors
        intrinsic_embeddings = self.intrinsic_encoder(financial_features, text_features)
        
        # Simple averaging of event features instead of RNN
        # Take most recent event only
        recent_events = event_sequences[:, 0, :]
        extrinsic_embeddings = self.extrinsic_encoder(recent_events)
        
        # Compute M&A intensity (timing module)
        intensity = self.timing_module(intrinsic_embeddings, extrinsic_embeddings, time_delta)
        
        # Apply choice module to compute target probabilities
        choice_scores = self.choice_module(intrinsic_embeddings, adj_matrix)
        
        # Apply available mask if provided
        if available_mask is not None:
            choice_scores = choice_scores * available_mask
            
        # Apply softmax to get probabilities
        choice_probs = F.softmax(choice_scores, dim=1)
        
        return intensity, choice_probs

# 5. Without GNN in choice module
class NoGNNModel(TDIN):
    def __init__(self, n_financial_features, n_text_features, event_dim, hidden_dim, embedding_dim):
        super().__init__(n_financial_features, n_text_features, event_dim, hidden_dim, embedding_dim)
        # Replace GNN with simple MLP
        self.choice_mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, financial_features, text_features, event_sequences, seq_lengths, 
                time_delta, adj_matrix, available_mask=None):
        # Encode intrinsic factors
        intrinsic_embeddings = self.intrinsic_encoder(financial_features, text_features)
        
        # Encode extrinsic factors
        extrinsic_embeddings = self.extrinsic_encoder(event_sequences, seq_lengths)
        
        # Compute M&A intensity (timing module)
        intensity = self.timing_module(intrinsic_embeddings, extrinsic_embeddings, time_delta)
        
        # Apply MLP instead of graph message passing
        node_embeddings = self.choice_mlp(intrinsic_embeddings)
        
        # Compute compatibility scores
        n_nodes = node_embeddings.shape[0]
        scores = torch.zeros((n_nodes, n_nodes), device=node_embeddings.device)
        
        for i in range(n_nodes):
            acquirer_embedding = node_embeddings[i].unsqueeze(0)
            # Simple dot product for compatibility
            compatibility = torch.matmul(acquirer_embedding, node_embeddings.t())
            scores[i] = compatibility.squeeze(0)
        
        # Apply available mask if provided
        if available_mask is not None:
            scores = scores * available_mask
            
        # Apply softmax to get probabilities
        choice_probs = F.softmax(scores, dim=1)
        
        return intensity, choice_probs

# Create and evaluate ablation models
print("\nRunning ablation study...")

# Initialize ablation models with same architecture but different behaviors
no_text_model = NoTextModel(n_features, n_text_features, 3, 64, 32).to(device)
no_text_model.load_state_dict(model.state_dict())  # Copy weights

no_network_model = NoDynamicNetworkModel(n_features, n_text_features, 3, 64, 32).to(device)
no_network_model.load_state_dict(model.state_dict())

no_peer_model = NoPeerEffectModel(n_features, n_text_features, 3, 64, 32).to(device)
no_peer_model.load_state_dict(model.state_dict())

no_rnn_model = NoRNNModel(n_features, n_text_features, 3, 64, 32).to(device)

no_gnn_model = NoGNNModel(n_features, n_text_features, 3, 64, 32).to(device)

# Run ablation tests
ablation_results = {
    "Full TDIN Model": final_auc,
    "Without Textual Embeddings": run_ablation_test("No Text", no_text_model),
    "Without Dynamic Industry Network": run_ablation_test("No Network", no_network_model),
    "Without Peer Effect Modeling": run_ablation_test("No Peer Effect", no_peer_model),
    "Without RNN in Timing Module": run_ablation_test("No RNN", no_rnn_model),
    "Without GNN in Choice Module": run_ablation_test("No GNN", no_gnn_model)
}

# Print ablation results
print("\nAblation Study Results:")
for ablation, auc in ablation_results.items():
    change = (auc - final_auc) / final_auc * 100
    print(f"{ablation}: AUC = {auc:.4f} (Change: {change:.2f}%)")

# Plot ablation results
plt.figure(figsize=(12, 8))
models = list(ablation_results.keys())
aucs = list(ablation_results.values())

bars = plt.bar(models, aucs, color='skyblue')
bars[0].set_color('green')  # Highlight full model

plt.axhline(y=baseline_auc, color='r', linestyle='--', label=f'Baseline (AUC = {baseline_auc:.4f})')
plt.ylim(min(aucs + [baseline_auc]) * 0.95, max(aucs) * 1.05)
plt.ylabel('AUC Score')
plt.title('Ablation Study Results')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('ablation_results.png')

print("\nSimulation complete. Results saved to files.")