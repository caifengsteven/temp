import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class StockDataset(Dataset):
    """Dataset for stock ranking"""
    def __init__(self, features, returns, n_stocks_per_period):
        self.features = features
        self.returns = returns
        self.n_stocks_per_period = n_stocks_per_period
        self.n_periods = len(features) // n_stocks_per_period
        
    def __len__(self):
        return self.n_periods
    
    def __getitem__(self, idx):
        start_idx = idx * self.n_stocks_per_period
        end_idx = start_idx + self.n_stocks_per_period
        
        period_features = self.features[start_idx:end_idx]
        period_returns = self.returns[start_idx:end_idx]
        
        # Get ranking based on returns (higher return = lower rank number)
        ranking = np.argsort(-period_returns)
        
        return torch.FloatTensor(period_features), torch.FloatTensor(period_returns), torch.LongTensor(ranking)

class ScoringNetwork(nn.Module):
    """Neural network for scoring stocks"""
    def __init__(self, input_dim, hidden_dims=[136, 272, 34]):
        super(ScoringNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # x shape: (batch_size, n_stocks, n_features)
        batch_size, n_stocks, n_features = x.shape
        
        # Reshape to process all stocks
        x = x.view(-1, n_features)
        scores = self.network(x)
        
        # Reshape back
        scores = scores.view(batch_size, n_stocks)
        
        return scores

class ListMLE(nn.Module):
    """ListMLE loss implementation"""
    def __init__(self, eps=1e-10):
        super(ListMLE, self).__init__()
        self.eps = eps
        
    def forward(self, scores, rankings):
        """
        scores: (batch_size, n_items) - predicted scores
        rankings: (batch_size, n_items) - ground truth rankings (0 = highest)
        """
        batch_size, n_items = scores.shape
        
        # Create permutation matrix from rankings
        device = scores.device
        perm_mat = torch.zeros(batch_size, n_items, n_items).to(device)
        
        for b in range(batch_size):
            for i in range(n_items):
                perm_mat[b, i, rankings[b, i]] = 1
        
        # Reorder scores according to ground truth ranking
        scores_sorted = torch.bmm(perm_mat, scores.unsqueeze(-1)).squeeze(-1)
        
        # Calculate ListMLE loss
        loss = 0
        for i in range(n_items):
            if i < n_items - 1:
                exp_scores = torch.exp(scores_sorted[:, i:])
                loss -= torch.log(torch.exp(scores_sorted[:, i]) / (torch.sum(exp_scores, dim=1) + self.eps))
        
        return torch.mean(loss)

class ListFold(nn.Module):
    """ListFold loss implementation"""
    def __init__(self, transform='exp', eps=1e-10):
        super(ListFold, self).__init__()
        self.transform = transform
        self.eps = eps
        
    def psi(self, x):
        """Transformation function"""
        if self.transform == 'exp':
            return torch.exp(torch.clamp(x, min=-10, max=10))  # Prevent overflow
        elif self.transform == 'sigmoid':
            return torch.sigmoid(x)
        else:
            return x
            
    def forward(self, scores, rankings):
        """
        scores: (batch_size, n_items) - predicted scores
        rankings: (batch_size, n_items) - ground truth rankings
        """
        batch_size, n_items = scores.shape
        
        # Ensure even number of items
        if n_items % 2 != 0:
            # If odd, ignore the last item
            scores = scores[:, :-1]
            rankings = rankings[:, :-1]
            n_items = n_items - 1
        
        n_pairs = n_items // 2
        
        # Create permutation matrix from rankings
        device = scores.device
        perm_mat = torch.zeros(batch_size, n_items, n_items).to(device)
        
        for b in range(batch_size):
            for i in range(n_items):
                perm_mat[b, i, rankings[b, i]] = 1
        
        # Reorder scores according to ground truth ranking
        scores_sorted = torch.bmm(perm_mat, scores.unsqueeze(-1)).squeeze(-1)
        
        loss = 0
        
        for i in range(n_pairs):
            # Score difference for the pair
            score_diff = scores_sorted[:, i] - scores_sorted[:, 2*n_pairs-1-i]
            
            # Calculate denominator: sum of all valid pairs
            denominator = self.eps
            for u in range(i, n_pairs):
                for v in range(u+1, 2*n_pairs-i):
                    if v <= 2*n_pairs-1-u:  # Valid pair check
                        pair_diff = scores_sorted[:, u] - scores_sorted[:, v]
                        denominator = denominator + self.psi(pair_diff)
            
            # Add to loss
            numerator = self.psi(score_diff)
            loss = loss - torch.log(numerator / denominator)
        
        return torch.mean(loss)

class MLPRegression(nn.Module):
    """Standard MLP for regression baseline"""
    def __init__(self):
        super(MLPRegression, self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, scores, returns):
        """MSE loss between predicted scores and actual returns"""
        # Average return across stocks for each batch
        return self.mse(scores.mean(dim=1), returns.mean(dim=1))

def generate_synthetic_data(n_periods=1000, n_stocks=80, n_factors=68):
    """Generate synthetic stock data with factors and returns"""
    
    # Generate factor loadings
    true_loadings = np.random.randn(n_factors) * 0.1
    
    # Generate features for each period
    all_features = []
    all_returns = []
    
    for period in range(n_periods):
        # Generate random factors for this period
        factors = np.random.randn(n_stocks, n_factors)
        
        # Add some cross-sectional patterns
        sector_effect = np.random.randn(n_stocks) * 0.02
        
        # Calculate returns
        signal = factors @ true_loadings
        noise = np.random.randn(n_stocks) * 0.05
        returns = signal + noise + sector_effect
        
        # Add momentum factor
        if period > 0 and len(all_returns) > 0:
            prev_returns = np.array(all_returns[-n_stocks:])
            momentum = prev_returns * 0.3
            returns = returns + momentum
        
        all_features.append(factors)
        all_returns.extend(returns)
    
    features = np.vstack(all_features)
    returns = np.array(all_returns)
    
    return features, returns

def evaluate_ranking(predicted_scores, true_returns, top_k=8):
    """Evaluate ranking performance"""
    n_stocks = len(predicted_scores)
    
    # Get rankings
    pred_ranking = np.argsort(-predicted_scores)
    true_ranking = np.argsort(-true_returns)
    
    # Calculate metrics
    # 1. Spearman correlation (IC)
    ic, _ = spearmanr(predicted_scores, true_returns)
    
    # 2. Long-short portfolio return
    long_stocks = pred_ranking[:top_k]
    short_stocks = pred_ranking[-top_k:]
    
    long_return = np.mean(true_returns[long_stocks])
    short_return = np.mean(true_returns[short_stocks])
    portfolio_return = long_return - short_return
    
    # 3. NDCG@k
    def dcg(ranking, returns, k):
        dcg_sum = 0
        for i in range(min(k, len(ranking))):
            stock_idx = ranking[i]
            # Normalize returns to [0, 1] for NDCG calculation
            normalized_rel = (returns[stock_idx] - np.min(returns)) / (np.max(returns) - np.min(returns) + 1e-10)
            dcg_sum += (2**normalized_rel - 1) / np.log2(i + 2)
        return dcg_sum
    
    actual_dcg = dcg(pred_ranking, true_returns, top_k)
    ideal_dcg = dcg(true_ranking, true_returns, top_k)
    ndcg = actual_dcg / (ideal_dcg + 1e-10)
    
    return {
        'ic': ic,
        'portfolio_return': portfolio_return,
        'long_return': long_return,
        'short_return': short_return,
        'ndcg': ndcg
    }

def train_model(model, loss_fn, train_loader, val_loader, n_epochs=50, lr=0.001):
    """Train the ranking model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for features, returns, rankings in train_loader:
            features = features.to(device)
            returns = returns.to(device)
            rankings = rankings.to(device)
            
            optimizer.zero_grad()
            scores = model(features)
            
            if isinstance(loss_fn, MLPRegression):
                loss = loss_fn(scores, returns)
            else:
                loss = loss_fn(scores, rankings)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for features, returns, rankings in val_loader:
                features = features.to(device)
                returns = returns.to(device)
                rankings = rankings.to(device)
                
                scores = model(features)
                
                if isinstance(loss_fn, MLPRegression):
                    loss = loss_fn(scores, returns)
                else:
                    loss = loss_fn(scores, rankings)
                
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return model, train_losses, val_losses

def backtest_strategy(model, test_loader, top_k=8, model_name='Model'):
    """Backtest the trading strategy"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    portfolio_returns = []
    ics = []
    long_returns = []
    short_returns = []
    
    with torch.no_grad():
        for features, returns, _ in test_loader:
            features = features.to(device)
            scores = model(features).cpu().numpy()
            returns = returns.numpy()
            
            # Evaluate each period
            for i in range(len(scores)):
                metrics = evaluate_ranking(scores[i], returns[i], top_k)
                portfolio_returns.append(metrics['portfolio_return'])
                ics.append(metrics['ic'])
                long_returns.append(metrics['long_return'])
                short_returns.append(metrics['short_return'])
    
    portfolio_returns = np.array(portfolio_returns)
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + portfolio_returns)
    
    # Calculate statistics
    annual_return = np.mean(portfolio_returns) * 52  # Weekly to annual
    annual_vol = np.std(portfolio_returns) * np.sqrt(52)
    sharpe_ratio = annual_return / (annual_vol + 1e-10)
    
    # Maximum drawdown
    rolling_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - rolling_max) / (rolling_max + 1e-10)
    max_drawdown = np.min(drawdown)
    
    print(f"\n{model_name} Performance:")
    print(f"Annual Return: {annual_return:.2%}")
    print(f"Annual Volatility: {annual_vol:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Average IC: {np.mean(ics):.3f}")
    print(f"Average Long Return: {np.mean(long_returns):.4f}")
    print(f"Average Short Return: {np.mean(short_returns):.4f}")
    
    return {
        'returns': portfolio_returns,
        'cumulative_returns': cumulative_returns,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'avg_ic': np.mean(ics)
    }

# Main execution
def main():
    # Generate synthetic data
    print("Generating synthetic data...")
    n_periods = 2000
    n_stocks = 80
    n_factors = 68
    
    features, returns = generate_synthetic_data(n_periods, n_stocks, n_factors)
    
    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Create datasets
    dataset = StockDataset(features, returns, n_stocks)
    
    # Split data
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize models and loss functions
    models = {
        'ListFold-exp': (ScoringNetwork(n_factors), ListFold(transform='exp')),
        'ListFold-sgm': (ScoringNetwork(n_factors), ListFold(transform='sigmoid')),
        'ListMLE': (ScoringNetwork(n_factors), ListMLE()),
        'MLP': (ScoringNetwork(n_factors), MLPRegression())
    }
    
    results = {}
    
    # Train and evaluate each model
    for model_name, (model, loss_fn) in models.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}...")
        print(f"{'='*50}")
        
        # Train model
        trained_model, train_losses, val_losses = train_model(
            model, loss_fn, train_loader, val_loader, n_epochs=50
        )
        
        # Backtest
        results[model_name] = backtest_strategy(trained_model, test_loader, model_name=model_name)
        
        # Plot training history
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{model_name} Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(results[model_name]['cumulative_returns'])
        plt.xlabel('Week')
        plt.ylabel('Cumulative Return')
        plt.title(f'{model_name} Backtest Performance')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Compare all models
    plt.figure(figsize=(12, 8))
    
    # Cumulative returns comparison
    plt.subplot(2, 2, 1)
    for model_name, result in results.items():
        plt.plot(result['cumulative_returns'], label=model_name)
    plt.xlabel('Week')
    plt.ylabel('Cumulative Return')
    plt.title('Cumulative Returns Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Sharpe ratio comparison
    plt.subplot(2, 2, 2)
    model_names = list(results.keys())
    sharpe_ratios = [results[name]['sharpe_ratio'] for name in model_names]
    colors = ['green' if 'ListFold' in name else 'blue' if name == 'ListMLE' else 'red' for name in model_names]
    plt.bar(model_names, sharpe_ratios, color=colors)
    plt.ylabel('Sharpe Ratio')
    plt.title('Sharpe Ratio Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Annual return comparison
    plt.subplot(2, 2, 3)
    annual_returns = [results[name]['annual_return'] for name in model_names]
    plt.bar(model_names, annual_returns, color=colors)
    plt.ylabel('Annual Return')
    plt.title('Annual Return Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # IC comparison
    plt.subplot(2, 2, 4)
    avg_ics = [results[name]['avg_ic'] for name in model_names]
    plt.bar(model_names, avg_ics, color=colors)
    plt.ylabel('Average IC')
    plt.title('Information Coefficient Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)
    print(f"{'Model':<15} {'Annual Return':<15} {'Sharpe Ratio':<15} {'Max Drawdown':<15} {'Avg IC':<10}")
    print("-"*80)
    
    for model_name, result in results.items():
        print(f"{model_name:<15} {result['annual_return']:>13.1%} {result['sharpe_ratio']:>14.2f} "
              f"{result['max_drawdown']:>14.1%} {result['avg_ic']:>9.3f}")
    
    # Additional analysis: Testing ListFold's long-short pair selection
    print("\n" + "="*80)
    print("TESTING LISTFOLD PAIR SELECTION")
    print("="*80)
    
    # Create a simple test case
    test_scores = torch.tensor([[5.0, 4.0, 1.0, 0.0]], requires_grad=True)
    test_rankings = torch.tensor([[0, 1, 2, 3]])
    
    listfold_exp = ListFold(transform='exp')
    loss = listfold_exp(test_scores, test_rankings)
    
    print(f"Test scores: {test_scores}")
    print(f"Test rankings: {test_rankings}")
    print(f"ListFold loss: {loss.item():.4f}")
    
    # Test with reversed ranking
    test_rankings_rev = torch.tensor([[3, 2, 1, 0]])
    loss_rev = listfold_exp(test_scores, test_rankings_rev)
    print(f"ListFold loss (reversed): {loss_rev.item():.4f}")
    print("Loss should be lower when ranking matches score order")

if __name__ == "__main__":
    main()