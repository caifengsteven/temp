import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.preprocessing import StandardScaler

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Simple neural network for asset allocation
class AssetAllocationNetwork(nn.Module):
    def __init__(self, input_size=3, hidden_size=10, output_size=2):
        super(AssetAllocationNetwork, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.softmax(self.output(x))
        return x

def main():
    # Set parameters
    num_assets = 2  # Stock and bond
    horizon_years = 30
    rebalance_freq = 1  # Annual rebalancing
    initial_wealth = 0
    annual_contribution = 10
    target_outperformance = 0.01  # 1% target outperformance
    num_periods = int(horizon_years / rebalance_freq)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set mean returns and covariance matrix based on historical data
    # Using estimates from the paper for stocks and bonds
    mean_returns = np.array([0.0666, 0.0083])  # Stock, Bond
    cov_matrix = np.array([
        [0.0218, 0.0008],  # Stock variance, Stock-Bond covariance
        [0.0008, 0.0011]   # Stock-Bond covariance, Bond variance
    ])
    
    # Generate training and testing data
    print("Generating simulated data...")
    num_train_paths = 10000
    num_test_paths = 1000
    
    # Generate returns for each rebalancing period
    train_returns = np.random.multivariate_normal(
        mean_returns * rebalance_freq,
        cov_matrix * rebalance_freq,
        size=(num_train_paths, num_periods)
    )
    
    test_returns = np.random.multivariate_normal(
        mean_returns * rebalance_freq,
        cov_matrix * rebalance_freq,
        size=(num_test_paths, num_periods)
    )
    
    # Define benchmark weights (50/50 constant proportion strategy)
    benchmark_weights = np.array([0.5, 0.5])  # 50% stocks, 50% bonds
    
    # Convert to torch tensors
    train_returns_tensor = torch.tensor(train_returns, dtype=torch.float32, device=device)
    benchmark_weights_tensor = torch.tensor(benchmark_weights, dtype=torch.float32, device=device)
    
    # Create the model
    model = AssetAllocationNetwork(input_size=3, hidden_size=10, output_size=num_assets).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Define asymmetric loss function
    def asymmetric_loss(adaptive_wealth, benchmark_wealth):
        # Calculate the elevated target
        elevated_target = benchmark_wealth * torch.exp(torch.tensor(target_outperformance * horizon_years, device=device))
        
        # Wealth difference
        wealth_diff = adaptive_wealth - elevated_target
        
        # Asymmetric penalty: quadratic for underperformance, linear for outperformance
        underperformance = torch.min(wealth_diff, torch.zeros_like(wealth_diff))
        overperformance = torch.max(wealth_diff, torch.zeros_like(wealth_diff))
        
        # Loss = min(W(T) - e^(sT) * W_b(T), 0)^2 + max(W(T) - e^(sT) * W_b(T), 0)
        loss = torch.mean(torch.square(underperformance) + overperformance)
        
        return loss
    
    # Define a simple training function
    def train_one_batch(batch_returns, batch_size=64):
        # Initialize wealth tensors
        adaptive_wealth = torch.zeros(batch_size, 1, device=device)
        benchmark_wealth = torch.zeros(batch_size, 1, device=device)
        
        # Set initial wealth
        adaptive_wealth.fill_(initial_wealth)
        benchmark_wealth.fill_(initial_wealth)
        
        # Simulate wealth trajectories
        for t in range(num_periods):
            # Add annual contribution
            adaptive_wealth = adaptive_wealth + annual_contribution
            benchmark_wealth = benchmark_wealth + annual_contribution
            
            # Prepare features for the model
            time_remaining = horizon_years - t * rebalance_freq
            time_remaining_tensor = torch.ones(batch_size, 1, device=device) * time_remaining
            
            features = torch.cat([
                adaptive_wealth,
                benchmark_wealth,
                time_remaining_tensor
            ], dim=1)
            
            # Get allocations from model
            allocations = model(features)
            
            # Calculate returns for the period
            period_returns = batch_returns[:, t, :]
            adaptive_period_return = torch.sum(allocations * period_returns, dim=1, keepdim=True)
            benchmark_period_return = torch.sum(benchmark_weights_tensor.expand(batch_size, -1) * period_returns, dim=1, keepdim=True)
            
            # Update wealth
            adaptive_wealth = adaptive_wealth * (1 + adaptive_period_return)
            benchmark_wealth = benchmark_wealth * (1 + benchmark_period_return)
        
        # Calculate loss
        return asymmetric_loss(adaptive_wealth.squeeze(), benchmark_wealth.squeeze())
    
    # Training loop
    print("Training model...")
    num_epochs = 50
    batch_size = 64
    num_batches = num_train_paths // batch_size
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        # Shuffle indices
        indices = torch.randperm(num_train_paths, device=device)
        
        for batch in range(num_batches):
            # Get batch indices
            batch_indices = indices[batch * batch_size:(batch + 1) * batch_size]
            
            # Get batch returns
            batch_returns = train_returns_tensor[batch_indices]
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass and compute loss
            loss = train_one_batch(batch_returns, batch_size)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Print epoch loss
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {epoch_loss/num_batches:.6f}")
    
    print("Model training completed.")
    
    # Evaluate on test data
    print("Evaluating the adaptive strategy...")
    
    # Convert test returns to tensor
    test_returns_tensor = torch.tensor(test_returns, dtype=torch.float32, device=device)
    
    # Evaluate function
    def evaluate_strategy(model, test_returns, benchmark_weights):
        with torch.no_grad():
            num_paths = test_returns.shape[0]
            
            # Initialize wealth arrays
            adaptive_wealth = np.zeros((num_paths, num_periods + 1))
            benchmark_wealth = np.zeros((num_paths, num_periods + 1))
            allocations = np.zeros((num_paths, num_periods, num_assets))
            
            # Set initial wealth
            adaptive_wealth[:, 0] = initial_wealth
            benchmark_wealth[:, 0] = initial_wealth
            
            # Simulate wealth trajectories
            for t in range(num_periods):
                # Add annual contribution
                adaptive_wealth[:, t] += annual_contribution
                benchmark_wealth[:, t] += annual_contribution
                
                # Prepare features for the model
                time_remaining = horizon_years - t * rebalance_freq
                features = np.column_stack((
                    adaptive_wealth[:, t],
                    benchmark_wealth[:, t],
                    np.ones(num_paths) * time_remaining
                ))
                
                # Convert to tensor
                features_tensor = torch.tensor(features, dtype=torch.float32, device=device)
                
                # Get allocations from model
                allocations[:, t] = model(features_tensor).cpu().numpy()
                
                # Calculate returns for the period
                adaptive_period_return = np.sum(allocations[:, t] * test_returns[:, t], axis=1)
                benchmark_period_return = np.sum(benchmark_weights * test_returns[:, t], axis=1)
                
                # Update wealth
                adaptive_wealth[:, t+1] = adaptive_wealth[:, t] * (1 + adaptive_period_return)
                benchmark_wealth[:, t+1] = benchmark_wealth[:, t] * (1 + benchmark_period_return)
            
            # Calculate metrics
            adaptive_terminal_wealth = adaptive_wealth[:, -1]
            benchmark_terminal_wealth = benchmark_wealth[:, -1]
            wealth_diff = adaptive_terminal_wealth - benchmark_terminal_wealth
            
            results = {
                'adaptive_mean': np.mean(adaptive_terminal_wealth),
                'adaptive_median': np.median(adaptive_terminal_wealth),
                'adaptive_std': np.std(adaptive_terminal_wealth),
                'benchmark_mean': np.mean(benchmark_terminal_wealth),
                'benchmark_median': np.median(benchmark_terminal_wealth),
                'benchmark_std': np.std(benchmark_terminal_wealth),
                'outperformance_prob': np.mean(wealth_diff > 0),
                'underperformance_prob': np.mean(wealth_diff < 0),
                'mean_wealth_diff': np.mean(wealth_diff),
                'median_wealth_diff': np.median(wealth_diff),
                'adaptive_wealth': adaptive_wealth,
                'benchmark_wealth': benchmark_wealth,
                'adaptive_allocations': allocations,
                'adaptive_terminal_wealth': adaptive_terminal_wealth,
                'benchmark_terminal_wealth': benchmark_terminal_wealth
            }
            
            return results
    
    # Evaluate adaptive strategy
    results = evaluate_strategy(model, test_returns, benchmark_weights)
    
    # Print results
    print("\nResults:")
    print(f"Adaptive Strategy Mean Terminal Wealth: {results['adaptive_mean']:.2f}")
    print(f"Adaptive Strategy Median Terminal Wealth: {results['adaptive_median']:.2f}")
    print(f"Adaptive Strategy Standard Deviation: {results['adaptive_std']:.2f}")
    print(f"Benchmark Strategy Mean Terminal Wealth: {results['benchmark_mean']:.2f}")
    print(f"Benchmark Strategy Median Terminal Wealth: {results['benchmark_median']:.2f}")
    print(f"Benchmark Strategy Standard Deviation: {results['benchmark_std']:.2f}")
    print(f"Probability of Outperforming Benchmark: {results['outperformance_prob']:.2%}")
    print(f"Mean Wealth Difference: {results['mean_wealth_diff']:.2f}")
    
    # Plot terminal wealth distribution
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    sns.histplot(results['adaptive_terminal_wealth'], kde=True, label='Adaptive Strategy', color='blue')
    sns.histplot(results['benchmark_terminal_wealth'], kde=True, label='Benchmark Strategy', color='red', alpha=0.7)
    plt.axvline(results['adaptive_median'], color='blue', linestyle='--', label=f'Adaptive Median: {results["adaptive_median"]:.0f}')
    plt.axvline(results['benchmark_median'], color='red', linestyle='--', label=f'Benchmark Median: {results["benchmark_median"]:.0f}')
    plt.title('Terminal Wealth Distribution')
    plt.xlabel('Terminal Wealth')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot CDF of terminal wealth
    plt.subplot(2, 2, 2)
    sns.ecdfplot(results['adaptive_terminal_wealth'], label='Adaptive Strategy', color='blue')
    sns.ecdfplot(results['benchmark_terminal_wealth'], label='Benchmark Strategy', color='red')
    plt.title('CDF of Terminal Wealth')
    plt.xlabel('Terminal Wealth')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    
    # Plot CDF of wealth difference
    plt.subplot(2, 2, 3)
    wealth_diff = results['adaptive_terminal_wealth'] - results['benchmark_terminal_wealth']
    sns.ecdfplot(wealth_diff, color='green')
    plt.axvline(0, color='black', linestyle='--')
    plt.title(f'CDF of Wealth Difference (Outperformance Prob: {results["outperformance_prob"]:.2%})')
    plt.xlabel('Wealth Difference (Adaptive - Benchmark)')
    plt.ylabel('Cumulative Probability')
    
    # Plot average asset allocation over time
    plt.subplot(2, 2, 4)
    mean_allocations = np.mean(results['adaptive_allocations'], axis=0)
    for i in range(num_assets):
        plt.plot(range(num_periods), mean_allocations[:, i], label=f'Asset {i+1}')
    plt.title('Average Asset Allocation Over Time')
    plt.xlabel('Period')
    plt.ylabel('Allocation')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Compare with 80/20 strategy
    print("\nComparing with 80/20 constant proportion strategy...")
    constant_8020_weights = np.array([0.8, 0.2])  # 80% stocks, 20% bonds
    
    # Evaluate 80/20 strategy
    def evaluate_constant_strategy(weights, test_returns, benchmark_weights):
        num_paths = test_returns.shape[0]
        
        # Initialize wealth arrays
        constant_wealth = np.zeros((num_paths, num_periods + 1))
        benchmark_wealth = np.zeros((num_paths, num_periods + 1))
        allocations = np.zeros((num_paths, num_periods, num_assets))
        
        # Set initial wealth
        constant_wealth[:, 0] = initial_wealth
        benchmark_wealth[:, 0] = initial_wealth
        
        # Simulate wealth trajectories
        for t in range(num_periods):
            # Add annual contribution
            constant_wealth[:, t] += annual_contribution
            benchmark_wealth[:, t] += annual_contribution
            
            # Set allocations to constant weights
            allocations[:, t] = weights
            
            # Calculate returns for the period
            constant_period_return = np.sum(allocations[:, t] * test_returns[:, t], axis=1)
            benchmark_period_return = np.sum(benchmark_weights * test_returns[:, t], axis=1)
            
            # Update wealth
            constant_wealth[:, t+1] = constant_wealth[:, t] * (1 + constant_period_return)
            benchmark_wealth[:, t+1] = benchmark_wealth[:, t] * (1 + benchmark_period_return)
        
        # Calculate metrics
        constant_terminal_wealth = constant_wealth[:, -1]
        benchmark_terminal_wealth = benchmark_wealth[:, -1]
        wealth_diff = constant_terminal_wealth - benchmark_terminal_wealth
        
        results = {
            'adaptive_mean': np.mean(constant_terminal_wealth),
            'adaptive_median': np.median(constant_terminal_wealth),
            'adaptive_std': np.std(constant_terminal_wealth),
            'benchmark_mean': np.mean(benchmark_terminal_wealth),
            'benchmark_median': np.median(benchmark_terminal_wealth),
            'benchmark_std': np.std(benchmark_terminal_wealth),
            'outperformance_prob': np.mean(wealth_diff > 0),
            'underperformance_prob': np.mean(wealth_diff < 0),
            'mean_wealth_diff': np.mean(wealth_diff),
            'median_wealth_diff': np.median(wealth_diff),
            'adaptive_wealth': constant_wealth,
            'benchmark_wealth': benchmark_wealth,
            'adaptive_allocations': allocations,
            'adaptive_terminal_wealth': constant_terminal_wealth,
            'benchmark_terminal_wealth': benchmark_terminal_wealth
        }
        
        return results
    
    # Evaluate 80/20 strategy
    results_8020 = evaluate_constant_strategy(constant_8020_weights, test_returns, benchmark_weights)
    
    # Print 80/20 results
    print(f"80/20 Strategy Mean Terminal Wealth: {results_8020['adaptive_mean']:.2f}")
    print(f"80/20 Strategy Median Terminal Wealth: {results_8020['adaptive_median']:.2f}")
    print(f"80/20 Strategy Standard Deviation: {results_8020['adaptive_std']:.2f}")
    print(f"Probability of 80/20 Outperforming Benchmark: {results_8020['outperformance_prob']:.2%}")
    
    # Compare adaptive and 80/20 strategies
    plt.figure(figsize=(15, 10))
    
    # Terminal wealth distribution comparison
    plt.subplot(2, 2, 1)
    sns.histplot(results['adaptive_terminal_wealth'], kde=True, label='Adaptive Strategy', color='blue')
    sns.histplot(results_8020['adaptive_terminal_wealth'], kde=True, label='80/20 Strategy', color='green', alpha=0.7)
    sns.histplot(results['benchmark_terminal_wealth'], kde=True, label='50/50 Benchmark', color='red', alpha=0.5)
    plt.title('Terminal Wealth Distribution Comparison')
    plt.xlabel('Terminal Wealth')
    plt.ylabel('Frequency')
    plt.legend()
    
    # CDF of terminal wealth comparison
    plt.subplot(2, 2, 2)
    sns.ecdfplot(results['adaptive_terminal_wealth'], label='Adaptive Strategy', color='blue')
    sns.ecdfplot(results_8020['adaptive_terminal_wealth'], label='80/20 Strategy', color='green')
    sns.ecdfplot(results['benchmark_terminal_wealth'], label='50/50 Benchmark', color='red')
    plt.title('CDF of Terminal Wealth')
    plt.xlabel('Terminal Wealth')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    
    # CDF of wealth difference comparison
    plt.subplot(2, 2, 3)
    wealth_diff_adaptive = results['adaptive_terminal_wealth'] - results['benchmark_terminal_wealth']
    wealth_diff_8020 = results_8020['adaptive_terminal_wealth'] - results['benchmark_terminal_wealth']
    
    sns.ecdfplot(wealth_diff_adaptive, label='Adaptive - Benchmark', color='blue')
    sns.ecdfplot(wealth_diff_8020, label='80/20 - Benchmark', color='green')
    plt.axvline(0, color='black', linestyle='--')
    plt.title('CDF of Wealth Difference')
    plt.xlabel('Wealth Difference')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    
    # Average allocation comparison
    plt.subplot(2, 2, 4)
    mean_allocations_adaptive = np.mean(results['adaptive_allocations'], axis=0)
    mean_allocations_8020 = np.mean(results_8020['adaptive_allocations'], axis=0)
    
    plt.plot(range(num_periods), mean_allocations_adaptive[:, 0], label='Adaptive Stock Allocation', color='blue')
    plt.plot(range(num_periods), mean_allocations_8020[:, 0], label='80/20 Stock Allocation', color='green')
    plt.axhline(y=0.5, color='red', linestyle='--', label='50/50 Benchmark Stock Allocation')
    plt.title('Average Stock Allocation Over Time')
    plt.xlabel('Period')
    plt.ylabel('Stock Allocation')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Test robustness with different market parameters
    print("\nTesting robustness with different market parameters...")
    
    # Generate data with lower stock returns (300 bps lower)
    mean_returns_low = np.array([0.0666 - 0.03, 0.0083])  # Reduced stock returns by 300 bps
    test_returns_low = np.random.multivariate_normal(
        mean_returns_low * rebalance_freq,
        cov_matrix * rebalance_freq,
        size=(num_test_paths, num_periods)
    )
    
    # Evaluate strategies on data with lower stock returns
    results_low = evaluate_strategy(model, test_returns_low, benchmark_weights)
    results_8020_low = evaluate_constant_strategy(constant_8020_weights, test_returns_low, benchmark_weights)
    
    # Print results for lower stock returns
    print("\nResults with Lower Stock Returns (300 bps lower):")
    print(f"Adaptive Strategy Mean Terminal Wealth: {results_low['adaptive_mean']:.2f}")
    print(f"Adaptive Strategy Median Terminal Wealth: {results_low['adaptive_median']:.2f}")
    print(f"Benchmark Strategy Mean Terminal Wealth: {results_low['benchmark_mean']:.2f}")
    print(f"Benchmark Strategy Median Terminal Wealth: {results_low['benchmark_median']:.2f}")
    print(f"Probability of Adaptive Outperforming Benchmark: {results_low['outperformance_prob']:.2%}")
    print(f"Probability of 80/20 Outperforming Benchmark: {results_8020_low['outperformance_prob']:.2%}")
    
    # Plot results for lower stock returns
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    sns.ecdfplot(results_low['adaptive_terminal_wealth'], label='Adaptive Strategy', color='blue')
    sns.ecdfplot(results_8020_low['adaptive_terminal_wealth'], label='80/20 Strategy', color='green')
    sns.ecdfplot(results_low['benchmark_terminal_wealth'], label='50/50 Benchmark', color='red')
    plt.title('CDF of Terminal Wealth with Lower Stock Returns')
    plt.xlabel('Terminal Wealth')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    wealth_diff_adaptive_low = results_low['adaptive_terminal_wealth'] - results_low['benchmark_terminal_wealth']
    wealth_diff_8020_low = results_8020_low['adaptive_terminal_wealth'] - results_low['benchmark_terminal_wealth']
    
    sns.ecdfplot(wealth_diff_adaptive_low, label='Adaptive - Benchmark', color='blue')
    sns.ecdfplot(wealth_diff_8020_low, label='80/20 - Benchmark', color='green')
    plt.axvline(0, color='black', linestyle='--')
    plt.title('CDF of Wealth Difference with Lower Stock Returns')
    plt.xlabel('Wealth Difference')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()