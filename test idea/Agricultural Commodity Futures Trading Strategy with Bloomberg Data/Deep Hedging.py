import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pdblp  # Python wrapper for Bloomberg API

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Connect to Bloomberg
try:
    con = pdblp.BCon(debug=False)
    con.start()
    BLOOMBERG_AVAILABLE = True
    print("Successfully connected to Bloomberg")
except:
    BLOOMBERG_AVAILABLE = False
    print("Bloomberg not available. Will use synthetic or historical data.")

class HedgingNetworkLayer(nn.Module):
    """
    Neural network layer for a single trading step in the Deep Hedging framework
    """
    def __init__(self, input_dim, hidden_units, output_dim):
        super(HedgingNetworkLayer, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_units[0]),
            nn.BatchNorm1d(hidden_units[0]),
            nn.ReLU(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.BatchNorm1d(hidden_units[1]),
            nn.ReLU(),
            nn.Linear(hidden_units[1], output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class DeepHedgingStrategy(nn.Module):
    def __init__(self, 
                 num_assets=1,
                 time_horizon=30,
                 trading_freq=1,  # in days
                 risk_aversion=0.5,
                 risk_measure='CVaR',
                 transaction_cost_proportion=0.0001,
                 hidden_units=[32, 32],
                 learning_rate=0.001):
        """
        Initialize the Deep Hedging Strategy
        
        Parameters:
        - num_assets: Number of hedging instruments
        - time_horizon: Time to maturity in days
        - trading_freq: Rebalancing frequency in days
        - risk_aversion: Parameter for risk measure (e.g., alpha for CVaR)
        - risk_measure: Risk measure to optimize ('CVaR', 'Entropy', 'MSE')
        - transaction_cost_proportion: Proportional transaction costs
        - hidden_units: List specifying number of neurons in hidden layers
        - learning_rate: Learning rate for Adam optimizer
        """
        super(DeepHedgingStrategy, self).__init__()
        
        self.num_assets = num_assets
        self.time_horizon = time_horizon
        self.trading_freq = trading_freq
        self.num_trading_steps = int(time_horizon / trading_freq)
        self.risk_aversion = risk_aversion
        self.risk_measure = risk_measure
        self.transaction_cost_proportion = transaction_cost_proportion
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        
        # Create a separate network for each time step
        # Each network takes [market_state, previous_position] as input
        # and outputs the new position
        self.hedging_networks = nn.ModuleList()
        
        # Market state dimension (price, possibly volatility, etc.)
        market_state_dim = num_assets + 1  # Adding 1 for potential additional features
        
        for t in range(self.num_trading_steps):
            # Input: market state + previous position
            input_dim = market_state_dim + num_assets
            
            # Create the network for this time step
            network = HedgingNetworkLayer(input_dim, hidden_units, num_assets)
            self.hedging_networks.append(network)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, market_paths):
        """
        Apply the hedging strategy to a batch of market paths
        
        Parameters:
        - market_paths: Tensor of shape [batch_size, num_steps+1, market_features]
          where market_features >= num_assets
        
        Returns:
        - portfolio_values: Terminal portfolio values
        - positions: Tensor of positions taken at each step
        - transaction_costs: Total transaction costs
        """
        batch_size = market_paths.shape[0]
        
        # Initialize positions (start with zero position)
        positions = torch.zeros(batch_size, self.num_trading_steps + 1, self.num_assets, device=device)
        
        # Track portfolio value (cash + position value)
        portfolio_values = torch.zeros(batch_size, self.num_trading_steps + 1, device=device)
        
        # Calculate transaction costs
        transaction_costs = torch.zeros(batch_size, device=device)
        
        for t in range(self.num_trading_steps):
            # Current market state
            market_state = market_paths[:, t, :]
            
            # Current position before trading
            prev_position = positions[:, t, :]
            
            # Network input: concatenate market state and previous position
            network_input = torch.cat([market_state, prev_position], dim=1)
            
            # Get new position from the network at this time step
            new_position = self.hedging_networks[t](network_input)
            positions[:, t+1, :] = new_position
            
            # Calculate price changes
            price_changes = market_paths[:, t+1, :self.num_assets] - market_paths[:, t, :self.num_assets]
            
            # Calculate P&L from hedging (dot product of position and price changes)
            pnl = torch.sum(prev_position * price_changes, dim=1)
            
            # Calculate transaction costs
            trade_sizes = torch.abs(new_position - prev_position)
            cost = self.transaction_cost_proportion * torch.sum(trade_sizes * market_paths[:, t, :self.num_assets], dim=1)
            transaction_costs += cost
            
            # Update portfolio value
            portfolio_values[:, t+1] = portfolio_values[:, t] + pnl - cost
        
        return portfolio_values[:, -1], positions, transaction_costs
    
    def compute_loss(self, terminal_portfolio_values, payoffs):
        """
        Compute the loss based on the specified risk measure
        
        Parameters:
        - terminal_portfolio_values: Portfolio values at terminal time
        - payoffs: Payoffs of the derivative to be hedged
        
        Returns:
        - loss: Risk measure of the hedging error
        """
        # Calculate hedging error
        hedging_error = terminal_portfolio_values - payoffs
        
        # Apply risk measure
        if self.risk_measure == 'CVaR':
            # Conditional Value at Risk (Expected Shortfall)
            sorted_errors = torch.sort(hedging_error)[0]
            cutoff_index = int(self.risk_aversion * len(sorted_errors))
            cvar = -torch.mean(sorted_errors[:cutoff_index])
            return cvar
            
        elif self.risk_measure == 'Entropy':
            # Entropic risk measure (negative of the expected utility)
            lambda_param = 1.0  # Risk aversion parameter
            return torch.mean(torch.exp(-lambda_param * hedging_error)) / lambda_param
            
        elif self.risk_measure == 'MSE':
            # Mean Squared Error
            return torch.mean(hedging_error**2)
            
        else:
            raise ValueError(f"Unsupported risk measure: {self.risk_measure}")

    def train_model(self, market_simulator, payoff_fn, num_epochs=100, batch_size=256, num_paths=10000, val_paths=1000):
        """
        Train the deep hedging model
        
        Parameters:
        - market_simulator: Function to generate market paths
        - payoff_fn: Function to calculate the payoff of the derivative
        - num_epochs: Number of training epochs
        - batch_size: Batch size for training
        - num_paths: Number of paths for training
        - val_paths: Number of paths for validation
        
        Returns:
        - training_losses: List of training losses per epoch
        - validation_losses: List of validation losses per epoch
        """
        # Generate training and validation paths
        train_paths = market_simulator(num_paths)
        val_paths = market_simulator(val_paths)
        
        # Calculate payoffs
        train_payoffs = payoff_fn(train_paths)
        val_payoffs = payoff_fn(val_paths)
        
        # Convert to PyTorch tensors
        train_paths_tensor = torch.tensor(train_paths, dtype=torch.float32, device=device)
        train_payoffs_tensor = torch.tensor(train_payoffs, dtype=torch.float32, device=device)
        val_paths_tensor = torch.tensor(val_paths, dtype=torch.float32, device=device)
        val_payoffs_tensor = torch.tensor(val_payoffs, dtype=torch.float32, device=device)
        
        # Create data loader
        train_dataset = TensorDataset(train_paths_tensor, train_payoffs_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        training_losses = []
        validation_losses = []
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            # Training loop
            self.train()  # Set to training mode
            for batch_paths, batch_payoffs in train_loader:
                self.optimizer.zero_grad()
                
                # Forward pass
                terminal_values, _, _ = self.forward(batch_paths)
                
                # Compute loss
                loss = self.compute_loss(terminal_values, batch_payoffs)
                
                # Backward pass
                loss.backward()
                
                # Update weights
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
            
            # Record training loss
            avg_train_loss = np.mean(epoch_losses)
            training_losses.append(avg_train_loss)
            
            # Validation
            self.eval()  # Set to evaluation mode
            with torch.no_grad():
                val_terminal_values, _, _ = self.forward(val_paths_tensor)
                val_loss = self.compute_loss(val_terminal_values, val_payoffs_tensor)
                validation_losses.append(val_loss.item())
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss.item():.4f}")
        
        return training_losses, validation_losses

    def get_price(self, market_simulator, payoff_fn, num_paths=10000):
        """
        Calculate the indifference price of the derivative
        
        Parameters:
        - market_simulator: Function to generate market paths
        - payoff_fn: Function to calculate the payoff of the derivative
        - num_paths: Number of paths for pricing
        
        Returns:
        - price: Indifference price of the derivative
        """
        # Generate paths for pricing
        pricing_paths = market_simulator(num_paths)
        
        # Calculate payoffs
        payoffs = payoff_fn(pricing_paths)
        
        # Convert to PyTorch tensors
        pricing_paths_tensor = torch.tensor(pricing_paths, dtype=torch.float32, device=device)
        payoffs_tensor = torch.tensor(payoffs, dtype=torch.float32, device=device)
        
        # Evaluate the strategy with the derivative
        self.eval()
        with torch.no_grad():
            terminal_values_with_deriv, _, _ = self.forward(pricing_paths_tensor)
            risk_with_deriv = self.compute_loss(terminal_values_with_deriv, payoffs_tensor).item()
            
            # Evaluate the strategy without the derivative (just zeros)
            zero_payoffs = torch.zeros_like(payoffs_tensor)
            risk_without_deriv = self.compute_loss(terminal_values_with_deriv, zero_payoffs).item()
        
        # The indifference price is the difference between the two risks
        price = risk_with_deriv - risk_without_deriv
        
        return price

    def test_strategy(self, market_paths, payoff):
        """
        Test the hedging strategy on new market paths
        
        Parameters:
        - market_paths: Market paths to test on
        - payoff: Payoffs of the derivative
        
        Returns:
        - hedging_errors: Hedging errors for each path
        - terminal_values: Terminal portfolio values
        - positions: Positions taken at each step
        - transaction_costs: Transaction costs
        """
        # Convert to PyTorch tensors if they aren't already
        if not isinstance(market_paths, torch.Tensor):
            market_paths = torch.tensor(market_paths, dtype=torch.float32, device=device)
        
        if not isinstance(payoff, torch.Tensor):
            payoff = torch.tensor(payoff, dtype=torch.float32, device=device)
        
        # Set to evaluation mode
        self.eval()
        
        # Forward pass
        with torch.no_grad():
            terminal_values, positions, transaction_costs = self.forward(market_paths)
            hedging_errors = terminal_values - payoff
        
        return hedging_errors, terminal_values, positions, transaction_costs

# Helper functions for market simulation and pricing

def generate_gbm_paths(S0, mu, sigma, T, n_steps, n_paths):
    """
    Generate geometric Brownian motion paths
    
    Parameters:
    - S0: Initial price
    - mu: Drift
    - sigma: Volatility
    - T: Time horizon
    - n_steps: Number of time steps
    - n_paths: Number of paths to generate
    
    Returns:
    - paths: Generated paths of shape [n_paths, n_steps+1, 2] 
             where the second dimension includes price and a placeholder for additional features
    """
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1, 2))  # Include room for price and additional features
    
    # Set initial price
    paths[:, 0, 0] = S0
    
    # Generate random shocks
    Z = np.random.normal(0, 1, size=(n_paths, n_steps))
    
    # Simulate paths
    for t in range(1, n_steps + 1):
        paths[:, t, 0] = paths[:, t-1, 0] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1])
    
    return paths

def generate_heston_paths(S0, v0, kappa, theta, xi, rho, T, n_steps, n_paths):
    """
    Generate paths from the Heston model
    
    Parameters:
    - S0: Initial price
    - v0: Initial variance
    - kappa: Rate of mean reversion
    - theta: Long-term variance
    - xi: Volatility of volatility
    - rho: Correlation between price and variance
    - T: Time horizon
    - n_steps: Number of time steps
    - n_paths: Number of paths to generate
    
    Returns:
    - paths: Generated paths of shape [n_paths, n_steps+1, 2]
             where the second dimension includes price and variance
    """
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1, 2))
    
    # Set initial values
    paths[:, 0, 0] = S0  # Price
    paths[:, 0, 1] = v0  # Variance
    
    # Generate correlated random shocks
    Z1 = np.random.normal(0, 1, size=(n_paths, n_steps))
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, size=(n_paths, n_steps))
    
    # Simulate paths
    for t in range(1, n_steps + 1):
        # Ensure variance stays positive
        paths[:, t-1, 1] = np.maximum(paths[:, t-1, 1], 0)
        
        # Update variance (using Euler scheme)
        paths[:, t, 1] = paths[:, t-1, 1] + kappa * (theta - paths[:, t-1, 1]) * dt + xi * np.sqrt(paths[:, t-1, 1] * dt) * Z1[:, t-1]
        
        # Update price
        paths[:, t, 0] = paths[:, t-1, 0] * np.exp(-0.5 * paths[:, t-1, 1] * dt + np.sqrt(paths[:, t-1, 1] * dt) * Z2[:, t-1])
    
    return paths

def call_option_payoff(paths, strike, idx=0):
    """
    Calculate payoff of a European call option
    
    Parameters:
    - paths: Market paths
    - strike: Strike price
    - idx: Index of the price in the path data (default is 0)
    
    Returns:
    - payoffs: Call option payoffs
    """
    terminal_prices = paths[:, -1, idx]
    return np.maximum(terminal_prices - strike, 0)

def put_option_payoff(paths, strike, idx=0):
    """
    Calculate payoff of a European put option
    
    Parameters:
    - paths: Market paths
    - strike: Strike price
    - idx: Index of the price in the path data (default is 0)
    
    Returns:
    - payoffs: Put option payoffs
    """
    terminal_prices = paths[:, -1, idx]
    return np.maximum(strike - terminal_prices, 0)

def get_historical_data_from_bloomberg(ticker, start_date, end_date, field='PX_LAST'):
    """
    Get historical price data from Bloomberg
    
    Parameters:
    - ticker: Bloomberg ticker
    - start_date: Start date in format 'YYYYMMDD'
    - end_date: End date in format 'YYYYMMDD'
    - field: Bloomberg field to retrieve (default is 'PX_LAST')
    
    Returns:
    - data: DataFrame with historical price data
    """
    if not BLOOMBERG_AVAILABLE:
        raise ValueError("Bloomberg connection is not available")
    
    data = con.bdh(ticker, field, start_date, end_date)
    return data

def prepare_historical_paths(data, n_steps, n_features=2):
    """
    Prepare historical data for use with the deep hedging model
    
    Parameters:
    - data: DataFrame with historical price data
    - n_steps: Number of time steps in each path
    - n_features: Number of features in each path
    
    Returns:
    - paths: Array of paths shaped for the deep hedging model
    """
    n_days = len(data)
    n_paths = n_days - n_steps
    
    paths = np.zeros((n_paths, n_steps + 1, n_features))
    
    for i in range(n_paths):
        window = data.iloc[i:i+n_steps+1].values
        paths[i, :, 0] = window.flatten()  # Price
        
        # Add other features if needed
        if n_features > 1:
            # For simplicity, just fill with zeros for now
            # In practice, you would add relevant features like implied volatility
            pass
    
    return paths

# Example usage

def example_heston_call_option():
    """
    Example: Hedging a call option in a Heston model with transaction costs
    """
    # Define market parameters
    S0 = 100.0       # Initial price
    v0 = 0.04        # Initial variance
    kappa = 1.0      # Rate of mean reversion
    theta = 0.04     # Long-term variance
    xi = 2.0         # Volatility of volatility
    rho = -0.7       # Correlation
    T = 30/365       # 30 days
    n_steps = 30     # Daily rebalancing
    
    # Option parameters
    strike = 100.0   # At-the-money call
    
    # Define market simulator
    def simulator(n_paths):
        return generate_heston_paths(S0, v0, kappa, theta, xi, rho, T, n_steps, n_paths)
    
    # Define payoff function
    def payoff_fn(paths):
        return call_option_payoff(paths, strike)
    
    # Create and train the deep hedging model
    model = DeepHedgingStrategy(
        num_assets=1,
        time_horizon=30,
        trading_freq=1,
        risk_aversion=0.5,
        risk_measure='CVaR',
        transaction_cost_proportion=0.0001,  # 1 basis point
        hidden_units=[32, 32],
        learning_rate=0.001
    ).to(device)
    
    # Train the model
    train_losses, val_losses = model.train_model(
        simulator, 
        payoff_fn, 
        num_epochs=100, 
        batch_size=256,
        num_paths=10000,
        val_paths=1000
    )
    
    # Calculate the price
    price = model.get_price(simulator, payoff_fn, num_paths=10000)
    print(f"Deep Hedging Price: {price:.4f}")
    
    # Test the strategy
    test_paths = simulator(1000)
    test_payoffs = payoff_fn(test_paths)
    
    hedging_errors, terminal_values, positions, transaction_costs = model.test_strategy(
        test_paths,
        test_payoffs
    )
    
    # Convert to numpy for analysis
    hedging_errors = hedging_errors.cpu().numpy()
    terminal_values = terminal_values.cpu().numpy()
    positions = positions.cpu().numpy()
    transaction_costs = transaction_costs.cpu().numpy()
    
    # Analyze results
    print(f"Mean Hedging Error: {np.mean(hedging_errors):.4f}")
    print(f"Std Dev of Hedging Error: {np.std(hedging_errors):.4f}")
    print(f"95% CVaR of Hedging Error: {np.mean(np.sort(hedging_errors)[:int(0.05*len(hedging_errors))]):.4f}")
    print(f"Mean Transaction Cost: {np.mean(transaction_costs):.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot training losses
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot histogram of hedging errors
    plt.subplot(2, 2, 2)
    plt.hist(hedging_errors, bins=50)
    plt.title('Distribution of Hedging Errors')
    plt.xlabel('Hedging Error')
    plt.ylabel('Frequency')
    
    # Plot a sample path and corresponding positions
    sample_idx = 0
    plt.subplot(2, 2, 3)
    plt.plot(test_paths[sample_idx, :, 0])
    plt.title('Sample Price Path')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    
    plt.subplot(2, 2, 4)
    plt.plot(positions[sample_idx, :, 0])
    plt.title('Hedging Positions for Sample Path')
    plt.xlabel('Time Step')
    plt.ylabel('Position')
    
    plt.tight_layout()
    plt.savefig('deep_hedging_results.png')
    plt.show()

def example_spx_call_option():
    """
    Example: Hedging a call option on the S&P 500 index using historical data
    """
    if not BLOOMBERG_AVAILABLE:
        print("Bloomberg not available. Using synthetic data instead.")
        # Use a GBM model as substitute
        example_heston_call_option()
        return
    
    # Get historical S&P 500 data
    spx_data = get_historical_data_from_bloomberg('SPX Index', '20180101', '20230101')
    
    # Prepare data
    time_horizon = 30  # 30 days
    paths = prepare_historical_paths(spx_data, time_horizon)
    
    # Define strike as the initial price of each path
    strikes = paths[:, 0, 0]
    
    def payoff_fn(paths):
        """Custom payoff function for historical paths"""
        return np.maximum(paths[:, -1, 0] - paths[:, 0, 0], 0)  # ATM call options
    
    # Create and train the deep hedging model
    model = DeepHedgingStrategy(
        num_assets=1,
        time_horizon=time_horizon,
        trading_freq=1,
        risk_aversion=0.75,
        risk_measure='CVaR',
        transaction_cost_proportion=0.0001,  # 1 basis point
        hidden_units=[32, 32],
        learning_rate=0.001
    ).to(device)
    
    # Split data into training and testing
    n_paths = len(paths)
    train_idx = int(0.8 * n_paths)
    train_paths = paths[:train_idx]
    test_paths = paths[train_idx:]
    
    # Create a simulator that samples from the training data
    def simulator(n_samples):
        indices = np.random.choice(len(train_paths), size=n_samples)
        return train_paths[indices]
    
    # Train the model
    train_losses, val_losses = model.train_model(
        simulator, 
        payoff_fn, 
        num_epochs=100, 
        batch_size=256,
        num_paths=len(train_paths),
        val_paths=min(1000, len(train_paths)//5)
    )
    
    # Test the strategy
    test_payoffs = payoff_fn(test_paths)
    
    hedging_errors, terminal_values, positions, transaction_costs = model.test_strategy(
        test_paths,
        test_payoffs
    )
    
    # Convert to numpy for analysis
    hedging_errors = hedging_errors.cpu().numpy()
    terminal_values = terminal_values.cpu().numpy()
    positions = positions.cpu().numpy()
    transaction_costs = transaction_costs.cpu().numpy()
    
    # Analyze results
    print(f"Mean Hedging Error: {np.mean(hedging_errors):.4f}")
    print(f"Std Dev of Hedging Error: {np.std(hedging_errors):.4f}")
    print(f"95% CVaR of Hedging Error: {np.mean(np.sort(hedging_errors)[:int(0.05*len(hedging_errors))]):.4f}")
    print(f"Mean Transaction Cost: {np.mean(transaction_costs):.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot training losses
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot histogram of hedging errors
    plt.subplot(2, 2, 2)
    plt.hist(hedging_errors, bins=50)
    plt.title('Distribution of Hedging Errors')
    plt.xlabel('Hedging Error')
    plt.ylabel('Frequency')
    
    # Plot a sample path and corresponding positions
    sample_idx = 0
    plt.subplot(2, 2, 3)
    plt.plot(test_paths[sample_idx, :, 0])
    plt.title('Sample Price Path')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    
    plt.subplot(2, 2, 4)
    plt.plot(positions[sample_idx, :, 0])
    plt.title('Hedging Positions for Sample Path')
    plt.xlabel('Time Step')
    plt.ylabel('Position')
    
    plt.tight_layout()
    plt.savefig('spx_deep_hedging_results.png')
    plt.show()

def compare_with_bs_delta():
    """
    Compare Deep Hedging with Black-Scholes delta hedging
    """
    # Define market parameters
    S0 = 100.0       # Initial price
    mu = 0.05        # Drift (not used in BS formula)
    sigma = 0.2      # Volatility
    T = 30/365       # 30 days
    n_steps = 30     # Daily rebalancing
    
    # Option parameters
    strike = 100.0   # At-the-money call
    
    # Define market simulator
    def simulator(n_paths):
        return generate_gbm_paths(S0, mu, sigma, T, n_steps, n_paths)
    
    # Define payoff function
    def payoff_fn(paths):
        return call_option_payoff(paths, strike)
    
    # Black-Scholes delta function
    def bs_delta(S, K, T, r, sigma):
        from scipy.stats import norm
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1)
    
    # Create and train the deep hedging model
    model = DeepHedgingStrategy(
        num_assets=1,
        time_horizon=30,
        trading_freq=1,
        risk_aversion=0.5,
        risk_measure='MSE',  # Use MSE for fair comparison with BS
        transaction_cost_proportion=0.0001,
        hidden_units=[32, 32],
        learning_rate=0.001
    ).to(device)
    
    # Train the model
    train_losses, val_losses = model.train_model(
        simulator, 
        payoff_fn, 
        num_epochs=100, 
        batch_size=256,
        num_paths=10000,
        val_paths=1000
    )
    
    # Generate test paths
    test_paths = simulator(1000)
    test_payoffs = payoff_fn(test_paths)
    
    # Test the deep hedging strategy
    dh_hedging_errors, dh_terminal_values, dh_positions, dh_transaction_costs = model.test_strategy(
        test_paths,
        test_payoffs
    )
    
    # Convert to numpy for analysis
    dh_hedging_errors = dh_hedging_errors.cpu().numpy()
    dh_positions = dh_positions.cpu().numpy()
    dh_transaction_costs = dh_transaction_costs.cpu().numpy()
    
    # Implement Black-Scholes delta hedging
    bs_positions = np.zeros((len(test_paths), n_steps + 1, 1))
    r = 0.0  # Risk-free rate (assumed zero for simplicity)
    
    for t in range(n_steps):
        remaining_T = (n_steps - t) / 365
        if remaining_T > 0:
            for i in range(len(test_paths)):
                S = test_paths[i, t, 0]
                bs_positions[i, t, 0] = bs_delta(S, strike, remaining_T, r, sigma)
    
    # Calculate BS hedging errors and transaction costs
    bs_pnl = np.zeros(len(test_paths))
    bs_transaction_costs = np.zeros(len(test_paths))
    
    for i in range(len(test_paths)):
        for t in range(n_steps):
            # P&L from price movement
            price_change = test_paths[i, t+1, 0] - test_paths[i, t, 0]
            bs_pnl[i] += bs_positions[i, t, 0] * price_change
            
            # Transaction costs
            if t > 0:
                trade_size = abs(bs_positions[i, t, 0] - bs_positions[i, t-1, 0])
                cost = 0.0001 * trade_size * test_paths[i, t, 0]
                bs_transaction_costs[i] += cost
                bs_pnl[i] -= cost
    
    bs_hedging_errors = bs_pnl - test_payoffs
    
    # Compare results
    print("Deep Hedging vs Black-Scholes Delta Hedging:")
    print(f"DH Mean Hedging Error: {np.mean(dh_hedging_errors):.4f}")
    print(f"BS Mean Hedging Error: {np.mean(bs_hedging_errors):.4f}")
    print(f"DH Std Dev of Hedging Error: {np.std(dh_hedging_errors):.4f}")
    print(f"BS Std Dev of Hedging Error: {np.std(bs_hedging_errors):.4f}")
    print(f"DH Mean Transaction Cost: {np.mean(dh_transaction_costs):.4f}")
    print(f"BS Mean Transaction Cost: {np.mean(bs_transaction_costs):.4f}")
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    # Plot training losses
    plt.subplot(2, 2, 1)
    plt.plot(train_losses)
    plt.title('Deep Hedging Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot histograms of hedging errors
    plt.subplot(2, 2, 2)
    plt.hist(dh_hedging_errors, bins=50, alpha=0.5, label='Deep Hedging')
    plt.hist(bs_hedging_errors, bins=50, alpha=0.5, label='Black-Scholes')
    plt.title('Distribution of Hedging Errors')
    plt.xlabel('Hedging Error')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot a sample path and corresponding positions
    sample_idx = 0
    plt.subplot(2, 2, 3)
    plt.plot(test_paths[sample_idx, :, 0])
    plt.title('Sample Price Path')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    
    plt.subplot(2, 2, 4)
    plt.plot(dh_positions[sample_idx, :, 0], label='Deep Hedging')
    plt.plot(bs_positions[sample_idx, :, 0], label='Black-Scholes')
    plt.title('Hedging Positions for Sample Path')
    plt.xlabel('Time Step')
    plt.ylabel('Position')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('hedging_comparison.png')
    plt.show()

# Run examples
if __name__ == "__main__":
    # Choose which example to run
    # example_heston_call_option()  # Heston model with call option
    # example_spx_call_option()     # S&P 500 with call option (requires Bloomberg)
    compare_with_bs_delta()       # Compare with Black-Scholes delta hedging