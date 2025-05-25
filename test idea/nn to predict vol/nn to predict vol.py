import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pdblp  # Python wrapper for Bloomberg API
import os
from scipy.stats import norm
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Create directories for saving models and results
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize Bloomberg connection
def connect_to_bloomberg():
    """
    Connect to Bloomberg API

    Returns:
    Bloomberg connection object or None if connection fails
    """
    try:
        con = pdblp.BCon(debug=True, port=8194)
        con.start()
        print("Connected to Bloomberg")
        return con
    except Exception as e:
        print(f"Failed to connect to Bloomberg: {e}")
        print("Using sample data instead of Bloomberg data")
        return None

# Calculate Black-Scholes delta
def black_scholes_delta(S, K, T, r, q, sigma):
    """
    Calculate Black-Scholes delta for a call option

    Parameters:
    S: Spot price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free rate
    q: Dividend yield
    sigma: Implied volatility

    Returns:
    Delta of the call option
    """
    d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return np.exp(-q * T) * norm.cdf(d1)

# Calculate Black-Scholes vega
def black_scholes_vega(S, K, T, r, q, sigma):
    """
    Calculate Black-Scholes vega for a call option

    Parameters:
    S: Spot price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free rate
    q: Dividend yield
    sigma: Implied volatility

    Returns:
    Vega of the option
    """
    d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

# Function to fetch S&P 500 data from Bloomberg
def get_sp500_data(con, start_date, end_date):
    """
    Fetch S&P 500 index data from Bloomberg

    Parameters:
    con: Bloomberg connection
    start_date: Start date for data retrieval (YYYY-MM-DD)
    end_date: End date for data retrieval (YYYY-MM-DD)

    Returns:
    DataFrame with S&P 500 daily data
    """
    if con is None:
        # Generate sample data if Bloomberg connection failed
        print("Using sample data for S&P 500")
        days = pd.date_range(start=start_date, end=end_date, freq='B')
        data = pd.DataFrame(index=days)

        # Generate synthetic price data
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.01, len(days))
        prices = [4000]  # Start with S&P at 4000
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        data['PX_LAST'] = prices[1:]

        # Calculate daily returns
        data['RETURN'] = data['PX_LAST'].pct_change().fillna(0)

        return data

    try:
        # Get S&P 500 price data
        spx_data = con.bdh("SPX Index", ["PX_LAST"], start_date, end_date)

        # Check if data was retrieved successfully
        if spx_data is None or 'PX_LAST' not in spx_data.columns:
            print("No valid S&P 500 data retrieved from Bloomberg")
            return get_sp500_data(None, start_date, end_date)

        # Calculate daily returns
        spx_data['RETURN'] = spx_data['PX_LAST'].pct_change().fillna(0)

        return spx_data
    except Exception as e:
        print(f"Error fetching S&P 500 data: {e}")
        return get_sp500_data(None, start_date, end_date)

# Function to fetch VIX data from Bloomberg
def get_vix_data(con, start_date, end_date):
    """
    Fetch VIX index data from Bloomberg

    Parameters:
    con: Bloomberg connection
    start_date: Start date for data retrieval (YYYY-MM-DD)
    end_date: End date for data retrieval (YYYY-MM-DD)

    Returns:
    DataFrame with VIX daily data
    """
    if con is None:
        # Generate sample data if Bloomberg connection failed
        print("Using sample data for VIX")
        days = pd.date_range(start=start_date, end=end_date, freq='B')
        data = pd.DataFrame(index=days)

        # Generate synthetic VIX data
        np.random.seed(43)
        data['VIX_LAST'] = np.random.normal(15, 3, len(days))
        data['VIX_LAST'] = data['VIX_LAST'].clip(9, 40)  # Clip to reasonable range

        return data

    try:
        # Get VIX data
        vix_data = con.bdh("VIX Index", ["PX_LAST"], start_date, end_date)

        # Check if data was retrieved successfully
        if vix_data is None or 'PX_LAST' not in vix_data.columns:
            print("No valid VIX data retrieved from Bloomberg")
            return get_vix_data(None, start_date, end_date)

        vix_data.columns = ['VIX_LAST']
        return vix_data
    except Exception as e:
        print(f"Error fetching VIX data: {e}")
        return get_vix_data(None, start_date, end_date)

# Function to fetch option data from Bloomberg
def get_option_data(con, start_date, end_date):
    """
    Fetch S&P 500 options data from Bloomberg

    Parameters:
    con: Bloomberg connection
    start_date: Start date for data retrieval (YYYY-MM-DD)
    end_date: End date for data retrieval (YYYY-MM-DD)

    Returns:
    DataFrame with option data
    """
    if con is None:
        # Generate sample data if Bloomberg connection failed
        print("Using sample data for options")

        # Generate dates and create empty dataframe
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        data = []

        # Generate synthetic S&P 500 data
        np.random.seed(42)
        spx_price = 4000
        for day in range(len(dates)):
            # Generate 20 options per day with different strikes and maturities
            for _ in range(20):
                # Generate option characteristics
                delta = np.random.uniform(0.05, 0.95)
                days_to_maturity = np.random.choice([30, 60, 90, 180, 270, 365])
                time_to_maturity = days_to_maturity / 365.0
                strike = spx_price * (1 + np.random.uniform(-0.2, 0.2))
                implied_vol = np.random.uniform(0.15, 0.35)

                # Create record for current day
                data.append({
                    'date': dates[day],
                    'option_id': f"OPTION_{day}_{_}",
                    'strike': strike,
                    'days_to_maturity': days_to_maturity,
                    'time_to_maturity': time_to_maturity,
                    'implied_vol': implied_vol,
                    'delta': delta,
                    'underlying_price': spx_price
                })

                # If not last day, create record for next day
                if day < len(dates) - 1:
                    # Update implied vol for next day
                    next_day_return = np.random.normal(0, 0.01)
                    spx_next = spx_price * (1 + next_day_return)

                    # Adjust implied vol based on underlying change
                    # Use a model similar to what's described in the paper:
                    # volatility response depends on delta, maturity, and return
                    vol_change = (-0.2 - 0.4 * delta + 0.5 * delta**2) * next_day_return / np.sqrt(time_to_maturity)
                    vol_change += np.random.normal(0, 0.005)  # Add some noise
                    next_implied_vol = implied_vol * (1 + vol_change)

                    # Make sure implied vol stays positive
                    next_implied_vol = max(0.05, next_implied_vol)

                    data.append({
                        'date': dates[day + 1],
                        'option_id': f"OPTION_{day}_{_}",
                        'strike': strike,
                        'days_to_maturity': max(0, days_to_maturity - 1),
                        'time_to_maturity': max(1/365.0, time_to_maturity - 1/365.0),
                        'implied_vol': next_implied_vol,
                        'delta': delta,  # Simplified - delta would actually change too
                        'underlying_price': spx_next
                    })

            # Update the underlying price for the next day
            spx_price = spx_price * (1 + np.random.normal(0.0005, 0.01))

        # Convert to DataFrame
        df = pd.DataFrame(data)
        return df

    try:
        # This is a placeholder for actual Bloomberg implementation
        # In practice, this would involve:
        # 1. Getting options chain using OMON function
        # 2. Fetching historical data for each option
        print("Fetching real options data from Bloomberg not implemented")
        return get_option_data(None, start_date, end_date)

    except Exception as e:
        print(f"Error fetching option data: {e}")
        return get_option_data(None, start_date, end_date)

# Process option data to create training dataset
def process_option_data(option_data, spx_data, vix_data):
    """
    Process option data to create features and target variable

    Parameters:
    option_data: DataFrame with option data
    spx_data: DataFrame with S&P 500 data
    vix_data: DataFrame with VIX data

    Returns:
    DataFrame with features and target
    """
    # Sort data by option ID and date
    option_data = option_data.sort_values(['option_id', 'date'])

    # Group by option ID
    grouped = option_data.groupby('option_id')

    # Initialize lists to store processed data
    processed_data = []

    # Loop through each option
    for _, group in grouped:
        # Skip if less than 2 days of data
        if len(group) < 2:
            continue

        # Create dataset with current and previous day data
        for i in range(1, len(group)):
            prev_day = group.iloc[i-1]
            current_day = group.iloc[i]

            # Skip if maturity less than 14 days or delta out of range
            if prev_day['days_to_maturity'] < 14 or prev_day['delta'] < 0.05 or prev_day['delta'] > 0.95:
                continue

            # Get date of previous day
            prev_date = prev_day['date']
            current_date = current_day['date']

            # Get S&P 500 return
            if prev_date in spx_data.index and current_date in spx_data.index:
                sp500_return = (spx_data.loc[current_date, 'PX_LAST'] / spx_data.loc[prev_date, 'PX_LAST']) - 1
            else:
                # Skip if S&P 500 data not available
                continue

            # Get VIX level (if using 4-feature model)
            if prev_date in vix_data.index:
                vix_level = vix_data.loc[prev_date, 'VIX_LAST'] / 100  # Convert to decimal
            else:
                # Skip if VIX data not available
                continue

            # Calculate implied volatility change
            implied_vol_change = current_day['implied_vol'] - prev_day['implied_vol']

            # Store data
            processed_data.append({
                'sp500_return': sp500_return,
                'time_to_maturity': prev_day['time_to_maturity'],
                'delta': prev_day['delta'],
                'vix_level': vix_level,
                'implied_vol_change': implied_vol_change
            })

    # Convert to DataFrame
    return pd.DataFrame(processed_data)

# Custom dataset class for PyTorch
class VolatilityDataset(Dataset):
    """
    Custom dataset for volatility surface modeling
    """
    def __init__(self, features, targets, use_vix=True):
        """
        Initialize dataset

        Parameters:
        features: DataFrame with features
        targets: Series with target variables
        use_vix: Whether to use VIX as a feature
        """
        self.features = features
        self.targets = targets
        self.use_vix = use_vix

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.use_vix:
            # Four-feature model
            x = torch.tensor([
                self.features.iloc[idx, 0],  # sp500_return
                self.features.iloc[idx, 1],  # time_to_maturity
                self.features.iloc[idx, 2],  # delta
                self.features.iloc[idx, 3]   # vix_level
            ], dtype=torch.float32)
        else:
            # Three-feature model
            x = torch.tensor([
                self.features.iloc[idx, 0],  # sp500_return
                self.features.iloc[idx, 1],  # time_to_maturity
                self.features.iloc[idx, 2]   # delta
            ], dtype=torch.float32)

        y = torch.tensor([self.targets.iloc[idx]], dtype=torch.float32)

        return x, y

# Neural Network model for volatility surface prediction
class VolatilityNN(nn.Module):
    """
    Neural network model for predicting implied volatility changes
    """
    def __init__(self, input_size, hidden_layers=3, nodes_per_layer=80):
        """
        Initialize the neural network

        Parameters:
        input_size: Number of input features
        hidden_layers: Number of hidden layers
        nodes_per_layer: Number of nodes per hidden layer
        """
        super(VolatilityNN, self).__init__()

        # Create a list to hold all layers
        layers = []

        # Input layer to first hidden layer
        layers.append(nn.Linear(input_size, nodes_per_layer))
        layers.append(nn.Sigmoid())

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(nodes_per_layer, nodes_per_layer))
            layers.append(nn.Sigmoid())

        # Output layer
        layers.append(nn.Linear(nodes_per_layer, 1))

        # Combine all layers into Sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass

        Parameters:
        x: Input tensor

        Returns:
        Predicted implied volatility change
        """
        return self.model(x)

# Function to train the model
def train_model(model, train_loader, val_loader, epochs=400, lr=0.001, weight_decay=1e-5):
    """
    Train the neural network model

    Parameters:
    model: Neural network model
    train_loader: DataLoader for training data
    val_loader: DataLoader for validation data
    epochs: Number of training epochs (default: 400, reduced from 4000 for faster debugging)
    lr: Learning rate
    weight_decay: L2 regularization parameter

    Returns:
    Trained model and training history
    """
    # Move model to device
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Initialize tracking variables
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update training loss
            train_loss += loss.item() * inputs.size(0)

        # Calculate average training loss
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                # Move data to device
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass
                outputs = model(inputs)

                # Calculate loss
                loss = criterion(outputs, targets)

                # Update validation loss
                val_loss += loss.item() * inputs.size(0)

        # Calculate average validation loss
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}')

    # Load best model
    model.load_state_dict(best_model_state)

    # Return trained model and history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }

    return model, history

# Function to evaluate model performance
def evaluate_model(model, test_loader):
    """
    Evaluate model performance on test data

    Parameters:
    model: Trained neural network model
    test_loader: DataLoader for test data

    Returns:
    Test loss (MSE) and predictions
    """
    # Move model to device
    model = model.to(device)

    # Set model to evaluation mode
    model.eval()

    # Initialize variables
    criterion = nn.MSELoss()
    test_loss = 0.0
    all_targets = []
    all_predictions = []

    # No gradient computation for evaluation
    with torch.no_grad():
        for inputs, targets in test_loader:
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Update test loss
            test_loss += loss.item() * inputs.size(0)

            # Store targets and predictions
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())

    # Calculate average test loss
    test_loss /= len(test_loader.dataset)

    return test_loss, np.array(all_targets), np.array(all_predictions)

# Function to calculate gain over benchmark
def calculate_gain(model_sse, benchmark_sse):
    """
    Calculate improvement gain over benchmark

    Parameters:
    model_sse: Sum of squared errors for the model
    benchmark_sse: Sum of squared errors for the benchmark

    Returns:
    Gain (improvement) as a percentage
    """
    gain = 1 - (model_sse / benchmark_sse)
    return gain * 100  # Convert to percentage

# Function to calculate analytic model predictions
def analytic_model_predictions(X, params):
    """
    Calculate predictions using the analytic model

    Parameters:
    X: Features (S&P 500 return, time to maturity, delta)
    params: Parameters (a, b, c) for the model

    Returns:
    Predicted implied volatility changes
    """
    sp500_return = X[:, 0]
    time_to_maturity = X[:, 1]
    delta = X[:, 2]

    a, b, c = params

    # Calculate predictions using the formula from the paper
    predictions = sp500_return * (a + b * delta + c * delta**2) / np.sqrt(time_to_maturity)

    return predictions.reshape(-1, 1)

# Function to estimate parameters for the analytic model
def fit_analytic_model(X, y):
    """
    Estimate parameters for the Hull-White analytic model

    Parameters:
    X: Features (S&P 500 return, time to maturity, delta)
    y: Target (implied volatility change)

    Returns:
    Estimated parameters (a, b, c) and model predictions
    """
    from scipy.optimize import minimize

    # Define objective function (sum of squared errors)
    def objective(params):
        predictions = analytic_model_predictions(X, params)
        return np.sum((predictions - y)**2)

    # Initial parameter guesses based on the paper
    initial_params = [-0.2329, 0.4176, -0.4892]

    # Optimize parameters
    result = minimize(objective, initial_params, method='BFGS')

    # Extract optimized parameters
    params = result.x

    # Calculate predictions
    predictions = analytic_model_predictions(X, params)

    return params, predictions

# Function to plot training history
def plot_training_history(history, model_name):
    """
    Plot training and validation losses

    Parameters:
    history: Dictionary with training history
    model_name: Name of the model for the plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_losses'], label='Training Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.title(f'Training and Validation Loss - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(f'results/{model_name}_training_history.png')
    plt.close()

# Function to plot volatility surface changes
def plot_volatility_surface(model, is_three_feature=True, vix_level=None):
    """
    Plot the volatility surface changes predicted by the model

    Parameters:
    model: Trained model (three or four feature)
    is_three_feature: Whether the model is three-feature or four-feature
    vix_level: VIX level for four-feature model (if None, use multiple levels)
    """
    # Set up grid for plotting
    deltas = np.linspace(0.1, 0.9, 9)  # Delta values from 0.1 to 0.9
    maturities = np.array([0.25, 0.5, 1.0, 1.5])  # Maturities in years

    # Index returns to plot
    returns = np.array([-0.0125, 0.0125])  # -1.25% and +1.25%
    return_labels = ["-1.25%", "+1.25%"]

    # VIX levels to plot (if four-feature model)
    if not is_three_feature and vix_level is None:
        vix_levels = np.array([0.13, 0.16])  # 13% and 16%
        vix_labels = ["VIX = 13%", "VIX = 16%"]
    elif not is_three_feature:
        vix_levels = np.array([vix_level])
        vix_labels = [f"VIX = {vix_level*100:.0f}%"]
    else:
        vix_levels = np.array([0])  # Dummy value
        vix_labels = [""]

    # Create meshgrid for 3D plotting
    delta_grid, maturity_grid = np.meshgrid(deltas, maturities)

    # Set model to evaluation mode
    model.eval()

    # Create figure with subplots
    num_plots = len(returns) * (len(vix_levels) if not is_three_feature else 1)
    fig = plt.figure(figsize=(15, 5 * num_plots))

    plot_idx = 1

    # Loop through combinations of returns and VIX levels
    for i, ret in enumerate(returns):
        for j, vix in enumerate(vix_levels):
            # Skip if three-feature model and not the first VIX level
            if is_three_feature and j > 0:
                continue

            # Create 3D subplot
            ax = fig.add_subplot(num_plots, 1, plot_idx, projection='3d')

            # Initialize array for volatility changes
            vol_changes = np.zeros_like(delta_grid)

            # Calculate predicted volatility changes for each point
            for m_idx, maturity in enumerate(maturities):
                for d_idx, delta in enumerate(deltas):
                    # Create input tensor
                    if is_three_feature:
                        x = torch.tensor([ret, maturity, delta], dtype=torch.float32).to(device)
                    else:
                        x = torch.tensor([ret, maturity, delta, vix], dtype=torch.float32).to(device)

                    # Get model prediction
                    with torch.no_grad():
                        vol_changes[m_idx, d_idx] = model(x).item()

            # Convert to basis points
            vol_changes = vol_changes * 10000

            # Create 3D surface plot
            surf = ax.plot_surface(delta_grid, maturity_grid, vol_changes, cmap=cm.coolwarm,
                                  linewidth=0, antialiased=True)

            # Set labels and title
            ax.set_xlabel('Delta')
            ax.set_ylabel('Time to Maturity (years)')
            ax.set_zlabel('Implied Volatility Change (bps)')

            if is_three_feature:
                ax.set_title(f'Implied Volatility Surface Change - Index Return: {return_labels[i]}')
            else:
                ax.set_title(f'Implied Volatility Surface Change - Index Return: {return_labels[i]}, {vix_labels[j]}')

            # Add color bar
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

            plot_idx += 1

    # Adjust layout and save figure
    plt.tight_layout()
    if is_three_feature:
        plt.savefig('results/three_feature_volatility_surface.png')
    else:
        if vix_level is None:
            plt.savefig('results/four_feature_volatility_surface.png')
        else:
            plt.savefig(f'results/four_feature_volatility_surface_vix_{int(vix_level*100)}.png')

    plt.close()

# Function to calculate minimum variance delta
def calculate_min_variance_delta(model, S, K, T, r, q, sigma, vix=None, delta_step=0.0001):
    """
    Calculate minimum variance delta using the model

    Parameters:
    model: Trained model (three or four feature)
    S: Spot price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free rate
    q: Dividend yield
    sigma: Implied volatility
    vix: VIX level (if using four-feature model)
    delta_step: Step size for finite difference approximation

    Returns:
    Minimum variance delta
    """
    # Calculate BS delta and vega
    bs_delta = black_scholes_delta(S, K, T, r, q, sigma)
    bs_vega = black_scholes_vega(S, K, T, r, q, sigma)

    # Calculate derivative of implied vol with respect to S
    # We'll use a simple finite difference approximation
    # Note: S_up and S_down are not directly used, but we calculate the returns
    # based on the percentage change

    # Calculate returns
    return_up = delta_step
    return_down = -delta_step

    # Prepare inputs for the model
    if vix is None:
        # Three-feature model
        x_up = torch.tensor([return_up, T, bs_delta], dtype=torch.float32).to(device)
        x_down = torch.tensor([return_down, T, bs_delta], dtype=torch.float32).to(device)
    else:
        # Four-feature model
        x_up = torch.tensor([return_up, T, bs_delta, vix], dtype=torch.float32).to(device)
        x_down = torch.tensor([return_down, T, bs_delta, vix], dtype=torch.float32).to(device)

    # Set model to evaluation mode
    model.eval()

    # Calculate predicted volatility changes
    with torch.no_grad():
        vol_change_up = model(x_up).item()
        vol_change_down = model(x_down).item()

    # Calculate derivative
    d_sigma_d_S = (vol_change_up - vol_change_down) / (2 * delta_step * S)

    # Calculate minimum variance delta
    min_var_delta = bs_delta + bs_vega * d_sigma_d_S

    return min_var_delta

# Main function to run the analysis
def run_volatility_analysis(start_date='2017-01-01', end_date='2020-12-31', test_model=False):
    """
    Run the complete volatility surface analysis

    Parameters:
    start_date: Start date for data collection
    end_date: End date for data collection
    test_model: Whether to test with pre-specified scenarios

    Returns:
    Trained models and evaluation results
    """
    # Step 1: Fetch data
    print("Connecting to Bloomberg...")
    con = connect_to_bloomberg()

    print(f"Fetching data from {start_date} to {end_date}...")
    sp500_data = get_sp500_data(con, start_date, end_date)
    vix_data = get_vix_data(con, start_date, end_date)
    option_data = get_option_data(con, start_date, end_date)

    # Step 2: Process data
    print("Processing option data...")
    processed_data = process_option_data(option_data, sp500_data, vix_data)

    # Print summary statistics
    print("\nSummary Statistics:")
    print(processed_data.describe())

    # Step 3: Split data into features and target
    X = processed_data[['sp500_return', 'time_to_maturity', 'delta', 'vix_level']]
    y = processed_data['implied_vol_change']

    # Step 4: Split data into training, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.22, random_state=42)

    print(f"Training data: {X_train.shape[0]} samples")
    print(f"Validation data: {X_val.shape[0]} samples")
    print(f"Test data: {X_test.shape[0]} samples")

    # Step 5: Fit the analytic model (benchmark)
    print("\nFitting the analytic model (benchmark)...")
    X_train_val_np = X_train_val[['sp500_return', 'time_to_maturity', 'delta']].values
    y_train_val_np = y_train_val.values.reshape(-1, 1)

    X_test_np = X_test[['sp500_return', 'time_to_maturity', 'delta']].values
    y_test_np = y_test.values.reshape(-1, 1)

    # Fit the analytic model and get parameters (predictions on training data not used)
    analytic_params, _ = fit_analytic_model(X_train_val_np, y_train_val_np)
    analytic_preds_test = analytic_model_predictions(X_test_np, analytic_params)

    print(f"Analytic model parameters: a={analytic_params[0]:.4f}, b={analytic_params[1]:.4f}, c={analytic_params[2]:.4f}")

    # Calculate analytic model MSE
    analytic_mse = np.mean((analytic_preds_test - y_test_np) ** 2)
    print(f"Analytic model test MSE: {analytic_mse:.8f}")

    # Step 6: Create PyTorch datasets
    train_dataset_3f = VolatilityDataset(X_train[['sp500_return', 'time_to_maturity', 'delta']], y_train, use_vix=False)
    val_dataset_3f = VolatilityDataset(X_val[['sp500_return', 'time_to_maturity', 'delta']], y_val, use_vix=False)
    test_dataset_3f = VolatilityDataset(X_test[['sp500_return', 'time_to_maturity', 'delta']], y_test, use_vix=False)

    train_dataset_4f = VolatilityDataset(X_train, y_train, use_vix=True)
    val_dataset_4f = VolatilityDataset(X_val, y_val, use_vix=True)
    test_dataset_4f = VolatilityDataset(X_test, y_test, use_vix=True)

    # Create DataLoaders
    train_loader_3f = DataLoader(train_dataset_3f, batch_size=512, shuffle=True)
    val_loader_3f = DataLoader(val_dataset_3f, batch_size=512)
    test_loader_3f = DataLoader(test_dataset_3f, batch_size=512)

    train_loader_4f = DataLoader(train_dataset_4f, batch_size=512, shuffle=True)
    val_loader_4f = DataLoader(val_dataset_4f, batch_size=512)
    test_loader_4f = DataLoader(test_dataset_4f, batch_size=512)

    # Step 7: Train and evaluate the three-feature model
    print("\nTraining the three-feature model...")
    model_3f = VolatilityNN(input_size=3, hidden_layers=3, nodes_per_layer=80)
    model_3f, history_3f = train_model(model_3f, train_loader_3f, val_loader_3f, epochs=400)

    # Save the model
    torch.save(model_3f.state_dict(), 'models/volatility_model_3f.pth')

    # Plot training history
    plot_training_history(history_3f, 'Three_Feature_Model')

    # Evaluate three-feature model
    test_loss_3f, _, _ = evaluate_model(model_3f, test_loader_3f)
    print(f"Three-feature model test MSE: {test_loss_3f:.8f}")

    # Calculate gain over analytic model
    gain_3f = calculate_gain(test_loss_3f, analytic_mse)
    print(f"Three-feature model gain over analytic model: {gain_3f:.2f}%")

    # Plot volatility surface for three-feature model
    plot_volatility_surface(model_3f, is_three_feature=True)

    # Step 8: Train and evaluate the four-feature model
    print("\nTraining the four-feature model...")
    model_4f = VolatilityNN(input_size=4, hidden_layers=3, nodes_per_layer=80)
    model_4f, history_4f = train_model(model_4f, train_loader_4f, val_loader_4f, epochs=400)

    # Save the model
    torch.save(model_4f.state_dict(), 'models/volatility_model_4f.pth')

    # Plot training history
    plot_training_history(history_4f, 'Four_Feature_Model')

    # Evaluate four-feature model
    test_loss_4f, _, _ = evaluate_model(model_4f, test_loader_4f)
    print(f"Four-feature model test MSE: {test_loss_4f:.8f}")

    # Calculate gain over three-feature model
    gain_4f = calculate_gain(test_loss_4f, test_loss_3f)
    print(f"Four-feature model gain over three-feature model: {gain_4f:.2f}%")

    # Plot volatility surface for four-feature model
    plot_volatility_surface(model_4f, is_three_feature=False)

    # Step 9: Test the model with specific scenarios if requested
    if test_model:
        print("\nTesting models with specific scenarios...")

        # Define test scenarios
        test_options = [
            {'S': 4000, 'K': 4000, 'T': 0.25, 'r': 0.03, 'q': 0.015, 'sigma': 0.15},  # ATM, 3-month
            {'S': 4000, 'K': 3800, 'T': 0.25, 'r': 0.03, 'q': 0.015, 'sigma': 0.18},  # ITM, 3-month
            {'S': 4000, 'K': 4200, 'T': 0.25, 'r': 0.03, 'q': 0.015, 'sigma': 0.20},  # OTM, 3-month
            {'S': 4000, 'K': 4000, 'T': 1.0, 'r': 0.03, 'q': 0.015, 'sigma': 0.17}    # ATM, 1-year
        ]

        # Test VIX levels
        vix_levels = [0.13, 0.16, 0.25]

        # Test return scenarios
        return_scenarios = [-0.0125, -0.005, 0.005, 0.0125]

        # Print header
        print("\nMin Variance Delta Estimates")
        print("=" * 80)
        print(f"{'Option':<10} {'Ret':<8} {'BS Delta':<10} {'Min-Var (3F)':<15} {'Min-Var (4F, VIX=13)':<20} {'Min-Var (4F, VIX=16)':<20} {'Min-Var (4F, VIX=25)':<20}")
        print("-" * 80)

        # Test each option and scenario
        for i, option in enumerate(test_options):
            for ret in return_scenarios:
                # Calculate BS delta
                bs_delta = black_scholes_delta(option['S'], option['K'], option['T'], option['r'], option['q'], option['sigma'])

                # Calculate min variance delta with three-feature model
                min_var_delta_3f = calculate_min_variance_delta(
                    model_3f, option['S'], option['K'], option['T'],
                    option['r'], option['q'], option['sigma']
                )

                # Calculate min variance delta with four-feature model for different VIX levels
                min_var_delta_4f = []
                for vix in vix_levels:
                    min_var_delta_4f.append(calculate_min_variance_delta(
                        model_4f, option['S'], option['K'], option['T'],
                        option['r'], option['q'], option['sigma'], vix
                    ))

                # Print results
                print(f"{i+1:<10} {ret*100:>6.2f}% {bs_delta:>10.4f} {min_var_delta_3f:>15.4f} {min_var_delta_4f[0]:>20.4f} {min_var_delta_4f[1]:>20.4f} {min_var_delta_4f[2]:>20.4f}")

        print("=" * 80)

    # Return models and results
    results = {
        'analytic_params': analytic_params,
        'analytic_mse': analytic_mse,
        'three_feature_model': model_3f,
        'three_feature_mse': test_loss_3f,
        'three_feature_gain': gain_3f,
        'four_feature_model': model_4f,
        'four_feature_mse': test_loss_4f,
        'four_feature_gain': gain_4f
    }

    return results

# Function to load a saved model and make real-time predictions
def real_time_predictions(model_path, use_vix=True, vix_level=None):
    """
    Make real-time predictions using a saved model

    Parameters:
    model_path: Path to the saved model
    use_vix: Whether to use VIX as a feature
    vix_level: Current VIX level (required if use_vix is True)

    Returns:
    Prediction function ready to use
    """
    # Load the model
    input_size = 4 if use_vix else 3
    model = VolatilityNN(input_size=input_size)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    if use_vix and vix_level is None:
        raise ValueError("VIX level must be provided when using a four-feature model")

    def predict_vol_change(index_return, time_to_maturity, delta):
        """
        Predict implied volatility change

        Parameters:
        index_return: S&P 500 index return
        time_to_maturity: Option time to maturity in years
        delta: Option delta

        Returns:
        Predicted implied volatility change
        """
        if use_vix:
            features = torch.tensor([index_return, time_to_maturity, delta, vix_level],
                                   dtype=torch.float32).to(device)
        else:
            features = torch.tensor([index_return, time_to_maturity, delta],
                                   dtype=torch.float32).to(device)

        with torch.no_grad():
            prediction = model(features).item()

        return prediction

    return predict_vol_change

# Function to run a live trading simulation
def run_live_simulation(model_3f_path='models/volatility_model_3f.pth',
                       model_4f_path='models/volatility_model_4f.pth'):
    """
    Run a simulation of using the models in a live trading environment

    Parameters:
    model_3f_path: Path to the saved three-feature model
    model_4f_path: Path to the saved four-feature model
    """
    print("Setting up live simulation...")

    # Connect to Bloomberg
    con = connect_to_bloomberg()

    # Get current S&P 500 and VIX values
    try:
        if con is not None:
            current_data = con.bdh("SPX Index", ["PX_LAST"],
                                   (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'),
                                   datetime.now().strftime('%Y-%m-%d'))
            vix_data = con.bdh("VIX Index", ["PX_LAST"],
                              (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'),
                              datetime.now().strftime('%Y-%m-%d'))

            spx_price = current_data['PX_LAST'].iloc[-1]
            vix_level = vix_data['PX_LAST'].iloc[-1] / 100  # Convert to decimal

            # Calculate daily return
            if len(current_data) > 1:
                spx_return = (current_data['PX_LAST'].iloc[-1] / current_data['PX_LAST'].iloc[-2]) - 1
            else:
                spx_return = 0
        else:
            # Use sample data
            spx_price = 4000
            vix_level = 0.15
            spx_return = 0.005
    except Exception as e:
        print(f"Error fetching live data: {e}")
        spx_price = 4000
        vix_level = 0.15
        spx_return = 0.005

    print(f"Current S&P 500: {spx_price:.2f}")
    print(f"Current VIX: {vix_level*100:.2f}%")
    print(f"Today's S&P 500 return: {spx_return*100:.2f}%")

    # Load models
    print("Loading models...")
    input_size_3f = 3
    input_size_4f = 4

    # Three-feature model
    model_3f = VolatilityNN(input_size=input_size_3f)
    model_3f.load_state_dict(torch.load(model_3f_path))
    model_3f.to(device)
    model_3f.eval()

    # Four-feature model
    model_4f = VolatilityNN(input_size=input_size_4f)
    model_4f.load_state_dict(torch.load(model_4f_path))
    model_4f.to(device)
    model_4f.eval()

    # Define sample options
    print("\nAnalyzing sample options...")
    sample_options = [
        {'name': 'ATM Call, 1m', 'K': spx_price, 'T': 1/12, 'sigma': 0.15, 'delta': 0.5},
        {'name': 'ITM Call, 3m', 'K': spx_price * 0.95, 'T': 0.25, 'sigma': 0.17, 'delta': 0.7},
        {'name': 'OTM Call, 3m', 'K': spx_price * 1.05, 'T': 0.25, 'sigma': 0.18, 'delta': 0.3},
        {'name': 'ATM Call, 1y', 'K': spx_price, 'T': 1.0, 'sigma': 0.20, 'delta': 0.5}
    ]

    # Calculate and display predicted volatility changes
    print("\nPredicted Volatility Changes (in basis points):")
    print(f"{'Option':<15} {'BS Model':<12} {'3-Feature':<12} {'4-Feature':<12} {'Difference':<12}")
    print("-" * 60)

    for option in sample_options:
        # Calculate BS model prediction
        a, b, c = -0.2329, 0.4176, -0.4892  # Default parameters from paper
        bs_prediction = spx_return * (a + b * option['delta'] + c * option['delta']**2) / np.sqrt(option['T'])

        # Three-feature model prediction
        features_3f = torch.tensor([spx_return, option['T'], option['delta']], dtype=torch.float32).to(device)
        with torch.no_grad():
            prediction_3f = model_3f(features_3f).item()

        # Four-feature model prediction
        features_4f = torch.tensor([spx_return, option['T'], option['delta'], vix_level], dtype=torch.float32).to(device)
        with torch.no_grad():
            prediction_4f = model_4f(features_4f).item()

        # Convert to basis points
        bs_prediction_bps = bs_prediction * 10000
        prediction_3f_bps = prediction_3f * 10000
        prediction_4f_bps = prediction_4f * 10000

        # Calculate difference between 3-feature and 4-feature
        difference = prediction_4f_bps - prediction_3f_bps

        # Print results
        print(f"{option['name']:<15} {bs_prediction_bps:>10.2f} {prediction_3f_bps:>10.2f} {prediction_4f_bps:>10.2f} {difference:>10.2f}")

    print("-" * 60)

    # Calculate and display minimum variance deltas
    print("\nMinimum Variance Delta Estimates:")
    print(f"{'Option':<15} {'BS Delta':<12} {'Min-Var (3F)':<15} {'Min-Var (4F)':<15} {'Difference':<12}")
    print("-" * 65)

    for option in sample_options:
        # Risk-free rate and dividend yield assumptions
        r = 0.03
        q = 0.015

        # Calculate BS delta
        bs_delta = option['delta']  # We already have this in our sample data

        # Calculate min variance delta with three-feature model
        min_var_delta_3f = calculate_min_variance_delta(
            model_3f, spx_price, option['K'], option['T'], r, q, option['sigma']
        )

        # Calculate min variance delta with four-feature model
        min_var_delta_4f = calculate_min_variance_delta(
            model_4f, spx_price, option['K'], option['T'], r, q, option['sigma'], vix_level
        )

        # Calculate difference
        difference = min_var_delta_4f - min_var_delta_3f

        # Print results
        print(f"{option['name']:<15} {bs_delta:>10.4f} {min_var_delta_3f:>15.4f} {min_var_delta_4f:>15.4f} {difference:>10.4f}")

    print("-" * 65)

if __name__ == "__main__":
    # Run the complete analysis with a smaller date range for faster debugging
    results = run_volatility_analysis(start_date='2020-01-01', end_date='2020-12-31', test_model=False)

    # Uncomment to run a live simulation after the model is trained
    # run_live_simulation()