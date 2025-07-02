import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

#########################################################################
# Part 1: Implement Generalized Affine Process Simulation
#########################################################################

def simulate_generalized_affine_path(x0, T, n_steps, params):
    """
    Simulate a path of a generalized affine process
    
    dXt = (b0 + b1*Xt) dt + (a0 + a1*Xt)^gamma dWt
    
    Parameters:
    - x0: initial value
    - T: time horizon
    - n_steps: number of time steps
    - params: dictionary containing b0, b1, a0, a1, gamma
    
    Returns:
    - t: time points
    - X: simulated path
    """
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    X = np.zeros(n_steps + 1)
    X[0] = x0
    
    # Extract parameters
    b0 = params['b0']
    b1 = params['b1']
    a0 = params['a0']
    a1 = params['a1']
    gamma = params['gamma']
    
    # Simulate path
    for i in range(n_steps):
        drift = (b0 + b1 * X[i]) * dt
        diffusion = (a0 + a1 * max(0, X[i]))**gamma * np.sqrt(dt) * np.random.normal()
        X[i+1] = X[i] + drift + diffusion
    
    return t, X

def simulate_robust_generalized_affine_path(x0, T, n_steps, param_ranges):
    """
    Simulate a path with parameter uncertainty
    
    Parameters:
    - x0: initial value
    - T: time horizon
    - n_steps: number of time steps
    - param_ranges: dictionary containing ranges for b0, b1, a0, a1, gamma
    
    Returns:
    - t: time points
    - X: simulated path
    - params_used: parameters used at each step
    """
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    X = np.zeros(n_steps + 1)
    X[0] = x0
    
    # Track parameters used
    params_used = {'b0': [], 'b1': [], 'a0': [], 'a1': [], 'gamma': []}
    
    # Simulate path with random parameters at each step
    for i in range(n_steps):
        # Sample parameters uniformly from ranges
        b0 = np.random.uniform(param_ranges['b0'][0], param_ranges['b0'][1])
        b1 = np.random.uniform(param_ranges['b1'][0], param_ranges['b1'][1])
        a0 = np.random.uniform(param_ranges['a0'][0], param_ranges['a0'][1])
        a1 = np.random.uniform(param_ranges['a1'][0], param_ranges['a1'][1])
        gamma = np.random.uniform(param_ranges['gamma'][0], param_ranges['gamma'][1])
        
        # Record parameters
        params_used['b0'].append(b0)
        params_used['b1'].append(b1)
        params_used['a0'].append(a0)
        params_used['a1'].append(a1)
        params_used['gamma'].append(gamma)
        
        # Simulate one step
        drift = (b0 + b1 * X[i]) * dt
        diffusion = (a0 + a1 * max(0, X[i]))**gamma * np.sqrt(dt) * np.random.normal()
        X[i+1] = X[i] + drift + diffusion
    
    return t, X, params_used

#########################################################################
# Part 2: Implement Payoff Functions for Different Derivatives
#########################################################################

def call_payoff(S, K):
    """
    Payoff function for a call option
    """
    return np.maximum(S - K, 0)

def put_payoff(S, K):
    """
    Payoff function for a put option
    """
    return np.maximum(K - S, 0)

def butterfly_payoff(S, K1, K2):
    """
    Payoff function for a butterfly option
    """
    return np.maximum(S - K1, 0) + np.maximum(S - K2, 0) - 2 * np.maximum(S - (K1 + K2) / 2, 0)

def lookback_call_payoff(path, K):
    """
    Payoff function for a lookback call option
    """
    return np.maximum(np.max(path) - K, 0)

def asian_put_payoff(path, K):
    """
    Payoff function for an Asian put option
    """
    return np.maximum(K - np.mean(path), 0)

#########################################################################
# Part 3: Implement Neural Network for Hedging Strategy
#########################################################################

class HedgingNetwork(nn.Module):
    """
    Neural network to represent hedging strategy
    
    Input: 
    - Current time t
    - Current stock price X_t
    - (Optional) Running maximum for path-dependent options
    
    Output:
    - Hedge ratio h_t
    """
    def __init__(self, input_dim=2, hidden_dim=64, n_layers=4):
        super(HedgingNetwork, self).__init__()
        
        # Create hidden layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, t, x, running_max=None):
        if running_max is None:
            inputs = torch.cat([t.unsqueeze(1), x.unsqueeze(1)], dim=1)
        else:
            inputs = torch.cat([t.unsqueeze(1), x.unsqueeze(1), running_max.unsqueeze(1)], dim=1)
        return self.net(inputs)

#########################################################################
# Part 4: Implement Robust Deep Hedging Algorithm
#########################################################################

class PathDataset(Dataset):
    """
    Dataset class for simulated paths
    """
    def __init__(self, paths, times):
        self.paths = paths
        self.times = times
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        return self.times, self.paths[idx]

def train_hedging_strategy(model, payoff_func, param_ranges, x0, T, n_steps, 
                          n_paths=10000, n_epochs=100, batch_size=256, 
                          learning_rate=0.001, path_dependent=False):
    """
    Train hedging strategy using Algorithm 1 from the paper
    
    Parameters:
    - model: neural network model
    - payoff_func: payoff function
    - param_ranges: parameter ranges for simulation
    - x0: initial value
    - T: time horizon
    - n_steps: number of time steps
    - n_paths: number of paths to simulate
    - n_epochs: number of training epochs
    - batch_size: batch size
    - learning_rate: learning rate
    
    Returns:
    - model: trained model
    - cash_position: initial cash position
    - training_losses: training loss history
    """
    # Initialize optimizer
    optimizer = optim.Adam(list(model.parameters()), lr=learning_rate)
    
    # Initialize cash position
    cash_position = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
    
    # Setup time points
    dt = T / n_steps
    times = torch.linspace(0, T, n_steps + 1)
    
    # Track training losses
    training_losses = []
    
    # Training loop
    for epoch in tqdm(range(n_epochs), desc="Training"):
        epoch_loss = 0
        
        # Generate new batch of paths for each epoch
        for batch in range(n_paths // batch_size):
            batch_paths = []
            payoffs = []
            
            # Generate paths with parameter uncertainty
            for _ in range(batch_size):
                _, path, _ = simulate_robust_generalized_affine_path(
                    x0, T, n_steps, param_ranges)
                batch_paths.append(path)
                payoffs.append(payoff_func(path))
            
            # Convert to torch tensors
            batch_paths = torch.tensor(batch_paths, dtype=torch.float32)
            payoffs = torch.tensor(payoffs, dtype=torch.float32)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Initialize hedging portfolio value
            portfolio_values = torch.zeros(batch_size, dtype=torch.float32)
            
            # Loop through time steps to compute hedging portfolio value
            for t in range(n_steps):
                current_time = times[t].repeat(batch_size)
                current_price = batch_paths[:, t]
                
                if path_dependent:
                    # Include running maximum for path-dependent options
                    running_max = torch.tensor([path[:t+1].max() for path in batch_paths], dtype=torch.float32)
                    hedge_ratio = model(current_time, current_price, running_max).squeeze()
                else:
                    hedge_ratio = model(current_time, current_price).squeeze()
                
                # Update portfolio value based on hedge
                price_change = batch_paths[:, t+1] - batch_paths[:, t]
                portfolio_values += hedge_ratio * price_change
            
            # Final portfolio value includes initial cash position
            portfolio_values += cash_position
            
            # Compute hedging error (squared difference)
            hedging_error = (portfolio_values - payoffs) ** 2
            loss = hedging_error.mean()
            
            # Backpropagate and update parameters
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Record average loss for this epoch
        avg_loss = epoch_loss / (n_paths // batch_size)
        training_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")
    
    return model, cash_position.item(), training_losses

#########################################################################
# Part 5: Evaluation Functions
#########################################################################

def evaluate_hedging_strategy(model, cash_position, payoff_func, param_ranges, x0, T, n_steps, 
                             n_paths=5000, path_dependent=False):
    """
    Evaluate hedging strategy on simulated paths
    
    Parameters:
    - model: trained neural network model
    - cash_position: initial cash position
    - payoff_func: payoff function
    - param_ranges: parameter ranges for simulation
    - x0: initial value
    - T: time horizon
    - n_steps: number of time steps
    - n_paths: number of paths to evaluate
    
    Returns:
    - rel_hedging_errors: relative hedging errors
    - hedge_values: final hedge values
    - payoffs: option payoffs
    """
    # Setup time points
    dt = T / n_steps
    times = torch.linspace(0, T, n_steps + 1)
    
    # Arrays to store results
    hedge_values = np.zeros(n_paths)
    payoffs = np.zeros(n_paths)
    
    # Evaluate on test paths
    model.eval()
    with torch.no_grad():
        for i in range(n_paths):
            # Generate path with parameter uncertainty
            _, path, _ = simulate_robust_generalized_affine_path(
                x0, T, n_steps, param_ranges)
            
            # Calculate option payoff
            payoff = payoff_func(path)
            payoffs[i] = payoff
            
            # Calculate hedging portfolio value
            portfolio_value = cash_position
            path_tensor = torch.tensor(path, dtype=torch.float32)
            
            for t in range(n_steps):
                current_time = times[t]
                current_price = path_tensor[t]
                
                if path_dependent:
                    running_max = torch.tensor(path[:t+1].max(), dtype=torch.float32)
                    hedge_ratio = model(current_time.unsqueeze(0), current_price.unsqueeze(0), 
                                       running_max.unsqueeze(0)).item()
                else:
                    hedge_ratio = model(current_time.unsqueeze(0), current_price.unsqueeze(0)).item()
                
                # Update portfolio value based on hedge
                price_change = path[t+1] - path[t]
                portfolio_value += hedge_ratio * price_change
            
            hedge_values[i] = portfolio_value
    
    # Calculate relative hedging errors
    hedging_errors = hedge_values - payoffs
    rel_hedging_errors = hedging_errors / np.maximum(np.abs(payoffs), 1e-6)
    
    return rel_hedging_errors, hedge_values, payoffs

def compare_hedging_strategies(robust_model, fixed_model, robust_cash, fixed_cash, 
                              payoff_func, param_ranges, fixed_params, x0, T, n_steps, 
                              n_paths=5000, path_dependent=False):
    """
    Compare robust hedging strategy with fixed-parameter hedging strategy
    
    Parameters:
    - robust_model: trained robust hedging model
    - fixed_model: trained fixed-parameter hedging model
    - robust_cash, fixed_cash: initial cash positions
    - payoff_func: payoff function
    - param_ranges: parameter ranges for robust evaluation
    - fixed_params: fixed parameters for comparison
    - x0, T, n_steps: simulation parameters
    - n_paths: number of paths to evaluate
    
    Returns:
    - robust_errors: relative hedging errors for robust strategy
    - fixed_errors: relative hedging errors for fixed-parameter strategy
    """
    # Evaluate robust strategy
    robust_errors, robust_values, payoffs = evaluate_hedging_strategy(
        robust_model, robust_cash, payoff_func, param_ranges, 
        x0, T, n_steps, n_paths, path_dependent)
    
    # Evaluate fixed-parameter strategy on the same paths with parameter uncertainty
    fixed_errors, fixed_values, _ = evaluate_hedging_strategy(
        fixed_model, fixed_cash, payoff_func, param_ranges, 
        x0, T, n_steps, n_paths, path_dependent)
    
    return robust_errors, fixed_errors, robust_values, fixed_values, payoffs

#########################################################################
# Part 6: Main Experiment Function
#########################################################################

def run_hedging_experiment(experiment_name, payoff_func, x0=10.0, T=30/365, n_steps=30,
                          param_ranges=None, fixed_params=None, path_dependent=False):
    """
    Run a hedging experiment comparing robust and fixed-parameter strategies
    
    Parameters:
    - experiment_name: name of the experiment
    - payoff_func: payoff function
    - x0, T, n_steps: simulation parameters
    - param_ranges: parameter ranges for robust simulation
    - fixed_params: fixed parameters for comparison
    
    Returns:
    - results: dictionary containing experiment results
    """
    # Default parameter ranges from the paper
    if param_ranges is None:
        param_ranges = {
            'a0': [0.3, 0.7],
            'a1': [0.4, 0.6],
            'b0': [-0.2, 0.2],
            'b1': [-0.1, 0.1],
            'gamma': [0.5, 1.5]
        }
    
    # Default fixed parameters (midpoints of ranges)
    if fixed_params is None:
        fixed_params = {
            'a0': 0.5,
            'a1': 0.5,
            'b0': 0.0,
            'b1': 0.0,
            'gamma': 1.0
        }
    
    print(f"\n{'='*80}")
    print(f"Running experiment: {experiment_name}")
    print(f"{'='*80}")
    
    # Initialize models
    input_dim = 3 if path_dependent else 2
    robust_model = HedgingNetwork(input_dim=input_dim)
    fixed_model = HedgingNetwork(input_dim=input_dim)
    
    # Train robust model
    print("\nTraining robust model...")
    robust_model, robust_cash, robust_losses = train_hedging_strategy(
        robust_model, payoff_func, param_ranges, x0, T, n_steps,
        n_paths=10000, n_epochs=100, path_dependent=path_dependent)
    
    # Train fixed-parameter model
    print("\nTraining fixed-parameter model...")
    
    # Create parameter ranges with zero width for fixed parameters
    fixed_ranges = {
        'a0': [fixed_params['a0'], fixed_params['a0']],
        'a1': [fixed_params['a1'], fixed_params['a1']],
        'b0': [fixed_params['b0'], fixed_params['b0']],
        'b1': [fixed_params['b1'], fixed_params['b1']],
        'gamma': [fixed_params['gamma'], fixed_params['gamma']]
    }
    
    fixed_model, fixed_cash, fixed_losses = train_hedging_strategy(
        fixed_model, payoff_func, fixed_ranges, x0, T, n_steps,
        n_paths=10000, n_epochs=100, path_dependent=path_dependent)
    
    # Compare strategies
    print("\nEvaluating strategies...")
    robust_errors, fixed_errors, robust_values, fixed_values, payoffs = compare_hedging_strategies(
        robust_model, fixed_model, robust_cash, fixed_cash,
        payoff_func, param_ranges, fixed_params, x0, T, n_steps,
        n_paths=5000, path_dependent=path_dependent)
    
    # Print results
    print("\nResults:")
    print(f"Mean absolute robust hedging error: {np.mean(np.abs(robust_errors)):.4f}")
    print(f"Mean absolute fixed hedging error: {np.mean(np.abs(fixed_errors)):.4f}")
    print(f"Std dev of robust hedging error: {np.std(robust_errors):.4f}")
    print(f"Std dev of fixed hedging error: {np.std(fixed_errors):.4f}")
    
    # Plot training losses
    plt.figure(figsize=(10, 6))
    plt.plot(robust_losses, label='Robust model')
    plt.plot(fixed_losses, label='Fixed-parameter model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss - {experiment_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'training_loss_{experiment_name.replace(" ", "_")}.png')
    plt.show()
    
    # Plot hedging errors
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.hist(robust_errors, bins=50, alpha=0.7, label='Robust')
    plt.hist(fixed_errors, bins=50, alpha=0.7, label='Fixed')
    plt.xlabel('Relative Hedging Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Hedging Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    error_diff = np.abs(fixed_errors) - np.abs(robust_errors)
    plt.hist(error_diff, bins=50)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Fixed Error - Robust Error')
    plt.ylabel('Frequency')
    plt.title('Difference in Absolute Hedging Errors')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.scatter(payoffs, robust_values, alpha=0.3, label='Robust')
    plt.scatter(payoffs, fixed_values, alpha=0.3, label='Fixed')
    plt.plot([min(payoffs), max(payoffs)], [min(payoffs), max(payoffs)], 'r--')
    plt.xlabel('Option Payoff')
    plt.ylabel('Hedge Value')
    plt.title('Hedge Value vs. Option Payoff')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Plot the hedging strategy for different prices
    x_range = np.linspace(0.8 * x0, 1.2 * x0, 100)
    t_mid = T / 2
    
    with torch.no_grad():
        robust_model.eval()
        fixed_model.eval()
        
        t_tensor = torch.tensor([t_mid] * len(x_range), dtype=torch.float32)
        x_tensor = torch.tensor(x_range, dtype=torch.float32)
        
        if path_dependent:
            # For simplicity, assume running max is 10% higher than current price
            running_max = torch.tensor(x_range * 1.1, dtype=torch.float32)
            robust_hedge = robust_model(t_tensor, x_tensor, running_max).numpy()
            fixed_hedge = fixed_model(t_tensor, x_tensor, running_max).numpy()
        else:
            robust_hedge = robust_model(t_tensor, x_tensor).numpy()
            fixed_hedge = fixed_model(t_tensor, x_tensor).numpy()
    
    plt.plot(x_range, robust_hedge, label='Robust')
    plt.plot(x_range, fixed_hedge, label='Fixed')
    plt.axvline(x=x0, color='k', linestyle='--', label='Initial price')
    plt.xlabel('Stock Price')
    plt.ylabel('Hedge Ratio')
    plt.title(f'Hedging Strategy at t={t_mid:.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'hedging_results_{experiment_name.replace(" ", "_")}.png')
    plt.show()
    
    # Return results
    results = {
        'experiment_name': experiment_name,
        'robust_model': robust_model,
        'fixed_model': fixed_model,
        'robust_cash': robust_cash,
        'fixed_cash': fixed_cash,
        'robust_errors': robust_errors,
        'fixed_errors': fixed_errors,
        'robust_values': robust_values,
        'fixed_values': fixed_values,
        'payoffs': payoffs,
        'robust_losses': robust_losses,
        'fixed_losses': fixed_losses
    }
    
    return results

#########################################################################
# Part 7: Run Experiments
#########################################################################

def main():
    # Define parameter settings
    x0 = 10.0  # Initial stock price
    T = 30/365  # Time horizon (30 days)
    n_steps = 30  # Number of time steps
    
    # Parameter ranges from the paper
    param_ranges = {
        'a0': [0.3, 0.7],
        'a1': [0.4, 0.6],
        'b0': [-0.2, 0.2],
        'b1': [-0.1, 0.1],
        'gamma': [0.5, 1.5]
    }
    
    # Fixed parameters (midpoints of ranges)
    fixed_params = {
        'a0': 0.5,
        'a1': 0.5,
        'b0': 0.0,
        'b1': 0.0,
        'gamma': 1.0
    }
    
    # Experiment 1: At-the-money call option
    call_func = lambda path: call_payoff(path[-1], x0)
    call_results = run_hedging_experiment(
        "ATM Call Option", call_func, x0, T, n_steps, param_ranges, fixed_params)
    
    # Experiment 2: Butterfly option
    butterfly_func = lambda path: butterfly_payoff(path[-1], 8, 12)
    butterfly_results = run_hedging_experiment(
        "Butterfly Option", butterfly_func, x0, T, n_steps, param_ranges, fixed_params)
    
    # Experiment 3: Lookback option
    lookback_func = lambda path: lookback_call_payoff(path, 12)
    lookback_results = run_hedging_experiment(
        "Lookback Call Option", lookback_func, x0, T, n_steps, param_ranges, fixed_params,
        path_dependent=True)
    
    # Experiment 4: Asian option
    asian_func = lambda path: asian_put_payoff(path, x0)
    asian_results = run_hedging_experiment(
        "Asian Put Option", asian_func, x0, T, n_steps, param_ranges, fixed_params)
    
    # Compare results across experiments
    experiments = [call_results, butterfly_results, lookback_results, asian_results]
    experiment_names = ["ATM Call", "Butterfly", "Lookback Call", "Asian Put"]
    
    robust_means = [np.mean(np.abs(exp['robust_errors'])) for exp in experiments]
    fixed_means = [np.mean(np.abs(exp['fixed_errors'])) for exp in experiments]
    robust_stds = [np.std(exp['robust_errors']) for exp in experiments]
    fixed_stds = [np.std(exp['fixed_errors']) for exp in experiments]
    
    # Create comparison table
    results_df = pd.DataFrame({
        'Experiment': experiment_names,
        'Robust Mean Error': robust_means,
        'Fixed Mean Error': fixed_means,
        'Robust Std Dev': robust_stds,
        'Fixed Std Dev': fixed_stds,
        'Improvement (%)': [(f - r) / f * 100 for f, r in zip(fixed_means, robust_means)]
    })
    
    print("\nComparison of results across experiments:")
    print(results_df)
    
    # Create bar chart comparing methods
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(len(experiment_names))
    
    plt.bar(index, fixed_means, bar_width, label='Fixed-parameter', alpha=0.8)
    plt.bar(index + bar_width, robust_means, bar_width, label='Robust', alpha=0.8)
    
    plt.xlabel('Experiment')
    plt.ylabel('Mean Absolute Hedging Error')
    plt.title('Comparison of Hedging Methods')
    plt.xticks(index + bar_width / 2, experiment_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('comparison_of_methods.png')
    plt.show()

if __name__ == "__main__":
    main()