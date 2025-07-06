import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class DeepPenaltyMethod:
    """
    Deep Penalty Method (DPM) for solving high dimensional optimal stopping problems.
    
    This implementation is based on the paper:
    "Deep Penalty Methods: A Class of Deep Learning Algorithms for Solving High Dimensional Optimal Stopping Problems"
    """
    
    def __init__(self, dim, T, N, r, mu, sigma, K, penalty_param, batch_size=64, hidden_layers=3, 
                 hidden_units=None, learning_rate=0.001):
        """
        Initialize the Deep Penalty Method
        
        Parameters:
        -----------
        dim : int
            Dimension of the problem (number of underlying assets)
        T : float
            Time horizon
        N : int
            Number of time steps
        r : float
            Risk-free interest rate
        mu : float
            Drift parameter
        sigma : float
            Volatility parameter
        K : float
            Strike price
        penalty_param : float
            Penalty parameter λ
        batch_size : int
            Batch size for training
        hidden_layers : int
            Number of hidden layers in the neural network
        hidden_units : int or None
            Number of units in each hidden layer. If None, will be set to dim + 10.
        learning_rate : float
            Learning rate for the optimizer
        """
        self.dim = dim
        self.T = T
        self.N = N
        self.dt = T / N
        self.r = r
        self.mu = mu
        self.sigma = sigma
        self.K = K
        self.penalty_param = penalty_param
        self.batch_size = batch_size
        
        # Neural network parameters
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units if hidden_units is not None else dim + 10
        self.learning_rate = learning_rate
        
        # Create the neural networks for each time step
        self.create_networks()
        
        # Trainable parameter for Y_0
        self.y0 = tf.Variable(0.0, dtype=tf.float32, name='y0')
        
    def create_networks(self):
        """Create neural networks for each time step."""
        self.networks = []
        
        for i in range(self.N):
            network = keras.Sequential([
                keras.layers.Dense(self.hidden_units, activation='relu', input_shape=(self.dim,)),
                keras.layers.Dense(self.hidden_units, activation='relu'),
                keras.layers.Dense(self.hidden_units, activation='relu'),
                keras.layers.Dense(self.dim)  # Output dimension = dim for Z
            ])
            self.networks.append(network)
    
    def stopping_payoff(self, x):
        """
        Compute the stopping payoff function p(t, x) = K - geometric mean of x
        
        Parameters:
        -----------
        x : tensor
            Asset prices
        
        Returns:
        --------
        tensor: Stopping payoff
        """
        # Geometric mean of the asset prices
        geo_mean = tf.exp(tf.reduce_mean(tf.math.log(x), axis=1, keepdims=True))
        return self.K - geo_mean
    
    def terminal_payoff(self, x):
        """
        Compute the terminal payoff function h(x) = max(K - geometric mean of x, 0)
        
        Parameters:
        -----------
        x : tensor
            Asset prices
        
        Returns:
        --------
        tensor: Terminal payoff
        """
        payoff = self.stopping_payoff(x)
        return tf.maximum(payoff, 0.0)
    
    def f1(self, t, x):
        """
        Compute f1(t, x) = (f(t,x) + Lp(t,x) - rp(t,x))e^(-rt)
        
        In this case, f(t,x) = 0, and we compute the other terms.
        
        Parameters:
        -----------
        t : float
            Time
        x : tensor
            Asset prices
        
        Returns:
        --------
        tensor: Value of f1(t,x)
        """
        # In this problem, f(t,x) = 0
        
        # Calculate Lp(t,x) - rp(t,x)
        # For the geometric mean payoff, this is a known expression
        # Lp(t,x) - rp(t,x) = p(t,x) * ((mu - r) - sigma^2 * (1 - 1/dim) / 2)
        
        # Geometric mean of the asset prices
        geo_mean = tf.exp(tf.reduce_mean(tf.math.log(x), axis=1, keepdims=True))
        p = self.K - geo_mean
        
        # Lp - rp term
        drift_adjustment = (self.mu - self.r)
        volatility_adjustment = -self.sigma**2 * (1 - 1/self.dim) / 2
        Lp_minus_rp = p * (drift_adjustment + volatility_adjustment)
        
        # Apply e^(-rt)
        return Lp_minus_rp * tf.exp(-self.r * t)
    
    def h1(self, x):
        """
        Compute h1(x) = (h(x) - p(T,x))e^(-rT)
        
        Parameters:
        -----------
        x : tensor
            Asset prices
        
        Returns:
        --------
        tensor: Value of h1(x)
        """
        terminal_payoff = self.terminal_payoff(x)
        stopping_payoff = self.stopping_payoff(x)
        return (terminal_payoff - stopping_payoff) * tf.exp(-self.r * self.T)
    
    def simulate_paths(self, batch_size):
        """
        Simulate asset price paths using Euler discretization
        
        Parameters:
        -----------
        batch_size : int
            Number of paths to simulate
        
        Returns:
        --------
        tensor: Simulated paths of shape [batch_size, N+1, dim]
        """
        dt = self.dt
        sqrt_dt = np.sqrt(dt)
        
        # Initialize paths with starting values of 1.0
        paths = np.zeros((batch_size, self.N + 1, self.dim))
        paths[:, 0, :] = 1.0
        
        # Simulate paths
        for i in range(self.N):
            # Generate random increments
            dW = np.random.normal(0, sqrt_dt, (batch_size, self.dim))
            
            # Update paths using Euler discretization
            paths[:, i+1, :] = paths[:, i, :] * np.exp(
                (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * dW
            )
        
        return tf.convert_to_tensor(paths, dtype=tf.float32)
    
    @tf.function
    def loss_fn(self, paths):
        """
        Compute the loss function for the DPM
        
        Parameters:
        -----------
        paths : tensor
            Simulated asset price paths
        
        Returns:
        --------
        tensor: Loss value
        """
        batch_size = tf.shape(paths)[0]
        
        # Initialize Y and Z
        y_values = tf.TensorArray(tf.float32, size=self.N+1)
        y_values = y_values.write(0, tf.ones((batch_size, 1)) * self.y0)
        
        # Forward propagation through time steps
        for i in range(self.N):
            t_i = i * self.dt
            x_i = paths[:, i, :]
            
            # Get Y value from previous step
            y_i = y_values.read(i)
            
            # Get Z value from neural network
            z_i = self.networks[i](x_i)
            
            # Generate Brownian increment
            dW = paths[:, i+1, :] - paths[:, i, :] - self.mu * paths[:, i, :] * self.dt
            dW = dW / (self.sigma * paths[:, i, :])
            
            # Penalty term: λ(max(-Y, 0))
            stopping_payoff = self.stopping_payoff(x_i)
            penalty_term = self.penalty_param * tf.maximum(-y_i - stopping_payoff * tf.exp(-self.r * t_i), 0.0)
            
            # Compute f1 term
            f1_term = self.f1(t_i, x_i)
            
            # Update Y using the BSDE
            y_i_plus_1 = y_i + (f1_term + penalty_term) * self.dt + tf.reduce_sum(z_i * dW, axis=1, keepdims=True)
            
            # Store Y value
            y_values = y_values.write(i+1, y_i_plus_1)
        
        # Terminal condition
        y_N = y_values.read(self.N)
        h1_xN = self.h1(paths[:, -1, :])
        
        # Compute mean squared error at terminal time
        loss = tf.reduce_mean(tf.square(y_N - h1_xN))
        
        return loss
    
    def train(self, epochs):
        """
        Train the DPM model
        
        Parameters:
        -----------
        epochs : int
            Number of training epochs
        
        Returns:
        --------
        list: Training loss history
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss_history = []
        
        # Create progress bar
        pbar = tqdm(range(epochs))
        
        # Training loop
        for epoch in pbar:
            # Simulate new paths
            paths = self.simulate_paths(self.batch_size)
            
            # Compute loss and gradients
            with tf.GradientTape() as tape:
                loss = self.loss_fn(paths)
            
            # Get trainable variables
            trainable_vars = [self.y0] + [var for network in self.networks for var in network.trainable_variables]
            
            # Compute and apply gradients
            gradients = tape.gradient(loss, trainable_vars)
            optimizer.apply_gradients(zip(gradients, trainable_vars))
            
            # Store loss
            loss_history.append(loss.numpy())
            
            # Update progress bar
            if epoch % 100 == 0:
                pbar.set_description(f"Epoch {epoch}, Loss: {loss.numpy():.6f}")
        
        return loss_history
    
    def price(self):
        """
        Return the estimated price of the American option
        
        Returns:
        --------
        float: Option price
        """
        # The option price is Y_0 + p(0, X_0)
        # Y_0 is stored in self.y0
        # p(0, X_0) = K - 1 since X_0 = 1 and geo_mean(X_0) = 1
        price = self.y0.numpy() * np.exp(self.r * 0) + (self.K - 1.0)
        return price
    
    def benchmark_price(self):
        """
        Compute the benchmark price using the one-dimensional reduction
        
        Returns:
        --------
        float: Benchmark option price
        """
        # For the American index put option, the problem can be reduced to a standard
        # one-dimensional American put option on the geometric average
        # We'll use a simple finite difference method for the benchmark
        
        # Adjusted parameters for the 1D problem
        mu_hat = self.mu - 0.5 * self.sigma**2 + 0.5/self.dim * self.sigma**2
        sigma_hat = self.sigma / np.sqrt(self.dim)
        
        # Grid parameters
        S_max = 3.0  # Maximum asset price
        M = 1000  # Number of price steps
        N = 1000  # Number of time steps
        
        # Create grids
        ds = S_max / M
        dt = self.T / N
        
        # Initialize grid
        grid = np.zeros((M+1, N+1))
        S_values = np.linspace(0, S_max, M+1)
        
        # Terminal condition (put payoff)
        for i in range(M+1):
            grid[i, N] = max(self.K - S_values[i], 0)
        
        # Backward induction
        for j in range(N-1, -1, -1):
            for i in range(1, M):
                # Finite difference coefficients
                a = 0.5 * dt * (sigma_hat**2 * i**2 - mu_hat * i)
                b = 1 - dt * (sigma_hat**2 * i**2 + self.r)
                c = 0.5 * dt * (sigma_hat**2 * i**2 + mu_hat * i)
                
                # Update grid point
                continuation = a * grid[i-1, j+1] + b * grid[i, j+1] + c * grid[i+1, j+1]
                exercise = self.K - S_values[i]
                grid[i, j] = max(continuation, exercise)
            
            # Boundary conditions
            grid[0, j] = self.K
            grid[M, j] = 0
        
        # Interpolate to get the price at S = 1
        idx = int(1.0 / ds)
        frac = (1.0 - idx * ds) / ds
        price = grid[idx, 0] * (1 - frac) + grid[idx+1, 0] * frac
        
        return price

# Test function to run the full experiment
def run_experiment(dim, epochs=10000):
    """
    Run the DPM experiment for a specific dimension
    
    Parameters:
    -----------
    dim : int
        Dimension of the problem
    epochs : int
        Number of training epochs
    
    Returns:
    --------
    dict: Results of the experiment
    """
    # Parameters from the paper
    T = 1.0
    N = 50
    r = 0.05
    mu = 0.05
    sigma = np.sqrt(2)
    K = 2.0
    penalty_param = 7.0  # λ = 700%
    batch_size = 5000
    
    # Start timer
    start_time = time.time()
    
    # Create and train the model
    model = DeepPenaltyMethod(dim=dim, T=T, N=N, r=r, mu=mu, sigma=sigma, K=K, 
                             penalty_param=penalty_param, batch_size=batch_size)
    
    loss_history = model.train(epochs)
    
    # Compute price and benchmark
    price = model.price()
    benchmark = model.benchmark_price()
    
    # Calculate variance of last 100 epochs
    variance = np.var(loss_history[-100:])
    
    # Calculate relative error
    relative_error = abs(price - benchmark) / benchmark * 100
    
    # Record running time
    running_time = time.time() - start_time
    
    # Return results
    results = {
        'dimension': dim,
        'price': price,
        'benchmark': benchmark,
        'variance': variance,
        'relative_error': relative_error,
        'training_loss': loss_history[-1],
        'running_time': running_time,
        'loss_history': loss_history
    }
    
    return results

# Run experiments for different dimensions
def run_all_experiments():
    """Run experiments for different dimensions and collect results"""
    dimensions = [10, 25, 50, 100]
    results = []
    
    for dim in dimensions:
        print(f"\nRunning experiment for dimension {dim}")
        result = run_experiment(dim)
        results.append(result)
        
        # Print results
        print(f"Price: {result['price']:.4f}")
        print(f"Benchmark: {result['benchmark']:.4f}")
        print(f"Relative Error: {result['relative_error']:.4f}%")
        print(f"Final Loss: {result['training_loss']:.6f}")
        print(f"Running Time: {result['running_time']:.2f}s")
    
    return results

# Create a table of results
def display_results_table(results):
    """Display results in a tabular format"""
    print("\n----- Results Table -----")
    print(f"{'Dim':>5} {'Price':>10} {'Benchmark':>10} {'Var (1e-8)':>12} {'Rel Error':>10} {'Loss':>10} {'Time (s)':>10}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['dimension']:>5} {result['price']:.4f} {result['benchmark']:.4f} "
              f"{result['variance']*1e8:.4f} {result['relative_error']:.4f}% "
              f"{result['training_loss']:.6f} {result['running_time']:.2f}")

# Plot loss history
def plot_loss_history(results):
    """Plot loss history for each dimension"""
    plt.figure(figsize=(10, 6))
    
    for result in results:
        plt.plot(result['loss_history'], label=f"Dim = {result['dimension']}")
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss by Dimension')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.show()

# Main execution
if __name__ == "__main__":
    # For demonstration purposes, we'll run a smaller experiment
    # to keep the execution time reasonable
    dim = 10
    epochs = 1000  # Reduced for demonstration
    
    print(f"Running experiment for dimension {dim} with {epochs} epochs...")
    result = run_experiment(dim, epochs)
    
    print("\n----- Results -----")
    print(f"Dimension: {result['dimension']}")
    print(f"Price: {result['price']:.4f}")
    print(f"Benchmark: {result['benchmark']:.4f}")
    print(f"Variance (last 100 epochs): {result['variance']*1e8:.6f}e-8")
    print(f"Relative Error: {result['relative_error']:.4f}%")
    print(f"Final Training Loss: {result['training_loss']:.6f}")
    print(f"Running Time: {result['running_time']:.2f}s")
    
    # Plot the loss history
    plt.figure(figsize=(10, 6))
    plt.plot(result['loss_history'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss (Dimension = {dim})')
    plt.yscale('log')
    plt.grid(True)
    plt.show()
    
    print("\nNote: For complete results matching the paper, run with dimensions [10, 25, 50, 100] and epochs=10000")