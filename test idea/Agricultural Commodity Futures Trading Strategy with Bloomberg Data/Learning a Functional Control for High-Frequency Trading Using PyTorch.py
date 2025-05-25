import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm
from scipy import stats
from sklearn.linear_model import LinearRegression

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

#=============================================================================
# 1. Optimal Execution Problem Simulation
#=============================================================================

class OptimalExecutionSimulator:
    """Simulator for the optimal execution problem with price impact"""
    
    def __init__(self, T=77, dt=1.0, initial_price=100.0, vol=0.1, 
                 alpha=0.1, kappa=0.1, A=0.01, phi=0.007, gamma=2.0,
                 with_seasonality=False):
        """
        Initialize the simulator
        
        Parameters:
        -----------
        T : int
            Number of time steps
        dt : float
            Time step size
        initial_price : float
            Initial price of the asset
        vol : float
            Volatility of the asset price
        alpha : float
            Permanent price impact coefficient
        kappa : float
            Temporary price impact coefficient
        A : float
            Terminal inventory penalty
        phi : float
            Running inventory penalty
        gamma : float
            Exponent for inventory penalty (2.0 for quadratic, 1.5 for sub-diffusive)
        with_seasonality : bool
            Whether to include intraday seasonality in the simulation
        """
        self.T = T
        self.dt = dt
        self.initial_price = initial_price
        self.vol = vol
        self.alpha = alpha
        self.kappa = kappa
        self.A = A
        self.phi = phi
        self.gamma = gamma
        self.with_seasonality = with_seasonality
        
        # Create seasonality patterns if needed
        if with_seasonality:
            # U-shaped volume profile
            self.volume_profile = 1.0 + 0.5 * (np.exp(-0.5 * ((np.arange(T) - 0) / 15) ** 2) + 
                                           0.8 * np.exp(-0.5 * ((np.arange(T) - (T-1)) / 15) ** 2))
            
            # Inverted U-shaped spread profile
            self.spread_profile = 1.0 + 0.5 * (np.exp(-0.5 * ((np.arange(T) - 0) / 15) ** 2) + 
                                          0.8 * np.exp(-0.5 * ((np.arange(T) - (T-1)) / 15) ** 2))
            
            # Normalize profiles
            self.volume_profile = self.volume_profile / np.mean(self.volume_profile)
            self.spread_profile = self.spread_profile / np.mean(self.spread_profile)
        
    def reset(self, initial_inventory=-100.0):
        """Reset the simulation with a given initial inventory"""
        self.t = 0
        self.inventory = initial_inventory
        self.price = self.initial_price
        self.wealth = 0.0
        
        self.price_path = [self.price]
        self.inventory_path = [self.inventory]
        self.wealth_path = [self.wealth]
        self.time_path = [self.t]
        
        return self._get_state()
    
    def _get_state(self):
        """Return the current state of the simulation"""
        return np.array([self.T - self.t, self.inventory])
    
    def step(self, trading_rate):
        """
        Take a step in the simulation given a trading rate
        
        Parameters:
        -----------
        trading_rate : float
            Trading rate (speed) for this step
            
        Returns:
        --------
        next_state : array
            Next state after taking the action
        reward : float
            Reward for this step
        done : bool
            Whether the episode is done
        info : dict
            Additional information
        """
        # Apply seasonality if needed
        if self.with_seasonality:
            vol_factor = self.volume_profile[self.t]
            spread_factor = self.spread_profile[self.t]
            curr_vol = self.vol * vol_factor
            curr_alpha = self.alpha * spread_factor / vol_factor
            curr_kappa = self.kappa * spread_factor / vol_factor
        else:
            curr_vol = self.vol
            curr_alpha = self.alpha
            curr_kappa = self.kappa
        
        # Increment time
        self.t += 1
        
        # Generate price noise
        price_noise = np.random.normal(0, curr_vol * np.sqrt(self.dt))
        
        # Update price, inventory, and wealth
        self.price += curr_alpha * trading_rate * self.dt + price_noise
        self.inventory += trading_rate * self.dt
        self.wealth -= trading_rate * (self.price + curr_kappa * trading_rate) * self.dt
        
        # Store path values
        self.price_path.append(self.price)
        self.inventory_path.append(self.inventory)
        self.wealth_path.append(self.wealth)
        self.time_path.append(self.t)
        
        # Check if the episode is done
        done = (self.t >= self.T)
        
        # Compute reward (negative cost for this step)
        running_penalty = -self.phi * np.abs(self.inventory) ** self.gamma * self.dt
        
        # Add terminal penalty if this is the final step
        terminal_penalty = 0
        if done:
            terminal_penalty = -self.A * np.abs(self.inventory) ** self.gamma
        
        reward = running_penalty + terminal_penalty
        
        return self._get_state(), reward, done, {}
    
    def simulate_trajectory(self, control_func, initial_inventory=-100.0):
        """
        Simulate a full trajectory using a control function
        
        Parameters:
        -----------
        control_func : callable
            Function that maps state to trading rate
        initial_inventory : float
            Initial inventory
            
        Returns:
        --------
        dict containing simulation results
        """
        state = self.reset(initial_inventory)
        done = False
        total_reward = 0
        trading_rates = []
        
        while not done:
            trading_rate = control_func(state)
            trading_rates.append(trading_rate)
            state, reward, done, _ = self.step(trading_rate)
            total_reward += reward
        
        # Add the final value of the position to the reward
        final_inventory_value = self.inventory_path[-1] * self.price_path[-1]
        final_reward = total_reward + final_inventory_value
        
        return {
            'price_path': np.array(self.price_path),
            'inventory_path': np.array(self.inventory_path),
            'wealth_path': np.array(self.wealth_path),
            'time_path': np.array(self.time_path),
            'trading_rates': np.array(trading_rates),
            'total_reward': total_reward,
            'final_reward': final_reward
        }

#=============================================================================
# 2. Neural Network Controller
#=============================================================================

class TradingNetwork(nn.Module):
    """Neural network for trading controller"""
    
    def __init__(self, input_dim=2, hidden_dims=[5, 5, 5], multi_preference=False):
        """
        Initialize the neural network
        
        Parameters:
        -----------
        input_dim : int
            Dimension of the input (2 for t and q)
        hidden_dims : list of int
            Dimensions of hidden layers
        multi_preference : bool
            Whether to include risk aversion parameters as inputs
        """
        super(TradingNetwork, self).__init__()
        
        # If multi-preference, add 2 more inputs for A and phi
        if multi_preference:
            input_dim += 2
            
        self.multi_preference = multi_preference
        
        # Create hidden layers
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(0.2))
            prev_dim = dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward pass"""
        return self.model(x)


class NeuralNetworkController:
    """Neural network controller for optimal execution"""
    
    def __init__(self, simulator, multi_preference=False, hidden_dims=[5, 5, 5],
                 learning_rate=5e-4, batch_size=64, device='cpu'):
        """
        Initialize the controller
        
        Parameters:
        -----------
        simulator : OptimalExecutionSimulator
            Simulator for the optimal execution problem
        multi_preference : bool
            Whether to include risk aversion parameters as inputs
        hidden_dims : list
            Dimensions of hidden layers
        learning_rate : float
            Learning rate for optimization
        batch_size : int
            Batch size for training
        device : str
            Device to use for training ('cpu' or 'cuda')
        """
        self.simulator = simulator
        self.multi_preference = multi_preference
        self.batch_size = batch_size
        self.device = device
        
        # Create neural network
        self.model = TradingNetwork(
            multi_preference=multi_preference,
            hidden_dims=hidden_dims
        ).to(device)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def control(self, state, A=None, phi=None):
        """
        Return trading rate for a given state
        
        Parameters:
        -----------
        state : array
            State of the system (T-t, q)
        A : float
            Terminal inventory penalty (only used in multi-preference mode)
        phi : float
            Running inventory penalty (only used in multi-preference mode)
            
        Returns:
        --------
        trading_rate : float
            Trading rate for the given state
        """
        # Convert to tensor
        if self.multi_preference:
            if A is None:
                A = self.simulator.A
            if phi is None:
                phi = self.simulator.phi
            state_tensor = torch.tensor(
                np.concatenate([state, [A, phi]]), 
                dtype=torch.float32
            ).to(self.device)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            trading_rate = self.model(state_tensor).item()
            
        return trading_rate
    
    def train_on_simulated_data(self, n_iterations=100000, n_validation=1000, 
                                initial_inventory=-100.0, tile_size=3, verbose=True):
        """
        Train the controller on simulated data
        
        Parameters:
        -----------
        n_iterations : int
            Number of SGD iterations
        n_validation : int
            Number of validation samples
        initial_inventory : float
            Initial inventory
        tile_size : int
            Number of inventory samples per Brownian path
        verbose : bool
            Whether to print progress
        
        Returns:
        --------
        dict containing training history
        """
        # Training history
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Use tqdm for progress bar if verbose
        iterator = tqdm(range(n_iterations)) if verbose else range(n_iterations)
        
        for i in iterator:
            # Training step
            loss = self._train_step(initial_inventory, tile_size)
            history['train_loss'].append(loss)
            
            # Validation step
            if (i + 1) % 100 == 0:
                val_loss = self._validate(n_validation, initial_inventory)
                history['val_loss'].append(val_loss)
                
                if verbose:
                    iterator.set_description(f"Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
        
        return history
    
    def _train_step(self, initial_inventory, tile_size):
        """
        Perform a single training step
        
        Parameters:
        -----------
        initial_inventory : float
            Initial inventory
        tile_size : int
            Number of inventory samples per Brownian path
            
        Returns:
        --------
        loss : float
            Training loss
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Create mini-batch
        total_loss = 0.0
        
        for _ in range(self.batch_size):
            # Reset simulator with different random seed
            self.simulator.reset(initial_inventory)
            
            # Generate different initial inventories for same Brownian path
            inventories = np.linspace(0.8, 1.2, tile_size) * initial_inventory
            
            for inventory in inventories:
                # For multi-preference, sample different preferences
                if self.multi_preference:
                    # Sample A and phi from a reasonable range
                    # defined in the paper to ensure exploration-exploitation
                    A = np.random.uniform(0.0001, 0.01)
                    phi = np.random.uniform(7e-5, 0.007)
                    
                    # Set simulator parameters
                    self.simulator.A = A
                    self.simulator.phi = phi
                
                # Reset simulator with this inventory
                state = self.simulator.reset(inventory)
                done = False
                episode_rewards = []
                
                # Simulate trajectory
                while not done:
                    # Get trading rate from current model
                    if self.multi_preference:
                        state_tensor = torch.tensor(
                            np.concatenate([state, [self.simulator.A, self.simulator.phi]]), 
                            dtype=torch.float32
                        ).to(self.device)
                    else:
                        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
                    
                    trading_rate_tensor = self.model(state_tensor)
                    trading_rate = trading_rate_tensor.item()
                    
                    # Take action in simulator
                    state, reward, done, _ = self.simulator.step(trading_rate)
                    episode_rewards.append(reward)
                
                # Calculate total reward for this episode
                total_reward = sum(episode_rewards)
                
                # Add negative reward as loss (we want to maximize reward)
                loss = -total_reward / (self.batch_size * tile_size)
                total_loss += loss

        # Create tensor for backward pass
        loss_tensor = torch.tensor(total_loss, dtype=torch.float32, requires_grad=True, device=self.device)
        
        # Backward pass
        loss_tensor.backward()
        self.optimizer.step()
        
        return total_loss
    
    def _validate(self, n_samples, initial_inventory):
        """
        Validate the model
        
        Parameters:
        -----------
        n_samples : int
            Number of validation samples
        initial_inventory : float
            Initial inventory
            
        Returns:
        --------
        val_loss : float
            Validation loss
        """
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for _ in range(n_samples):
                # For multi-preference, sample different preferences
                if self.multi_preference:
                    # Sample A and phi from range
                    A = np.random.uniform(0.0001, 0.01)
                    phi = np.random.uniform(7e-5, 0.007)
                    
                    # Set simulator parameters
                    self.simulator.A = A
                    self.simulator.phi = phi
                
                # Define control function
                def control_func(state):
                    return self.control(state, A if self.multi_preference else None, 
                                        phi if self.multi_preference else None)
                
                # Simulate trajectory
                trajectory = self.simulator.simulate_trajectory(
                    control_func, initial_inventory
                )
                
                val_loss -= trajectory['total_reward'] / n_samples
        
        return val_loss
    
    def train_on_real_data(self, real_data, n_iterations=10000, verbose=True):
        """
        Train the controller on real data using transfer learning
        
        Parameters:
        -----------
        real_data : list of dict
            List of real trajectories
        n_iterations : int
            Number of SGD iterations
        verbose : bool
            Whether to print progress
            
        Returns:
        --------
        dict containing training history
        """
        # Training history
        history = {
            'train_loss': []
        }
        
        # Use tqdm for progress bar if verbose
        iterator = tqdm(range(n_iterations)) if verbose else range(n_iterations)
        
        for i in iterator:
            # Training step
            loss = self._train_step_real_data(real_data)
            history['train_loss'].append(loss)
            
            if verbose and (i + 1) % 100 == 0:
                iterator.set_description(f"Loss: {loss:.4f}")
        
        return history
    
    def _train_step_real_data(self, real_data):
        """
        Perform a single training step on real data
        
        Parameters:
        -----------
        real_data : list of dict
            List of real trajectories
            
        Returns:
        --------
        loss : float
            Training loss
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Create mini-batch by sampling trajectories
        total_loss = 0.0
        
        for _ in range(self.batch_size):
            # Sample a random trajectory
            trajectory = real_data[np.random.randint(len(real_data))]
            
            # Track rewards for this trajectory
            total_reward = 0
            
            # Iterate through the trajectory
            for t in range(len(trajectory['time_path']) - 1):
                # Get state
                state = np.array([self.simulator.T - trajectory['time_path'][t], 
                                 trajectory['inventory_path'][t]])
                
                # For multi-preference, add A and phi
                if self.multi_preference:
                    state_tensor = torch.tensor(
                        np.concatenate([state, [self.simulator.A, self.simulator.phi]]), 
                        dtype=torch.float32
                    ).to(self.device)
                else:
                    state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
                
                # Get trading rate from model
                _ = self.model(state_tensor)
                
                # Calculate reward
                inventory = trajectory['inventory_path'][t]
                next_inventory = trajectory['inventory_path'][t+1]
                
                # Running penalty
                running_penalty = -self.simulator.phi * np.abs(inventory) ** self.simulator.gamma * self.simulator.dt
                
                # Terminal penalty if at end
                terminal_penalty = 0
                if t == len(trajectory['time_path']) - 2:
                    terminal_penalty = -self.simulator.A * np.abs(next_inventory) ** self.simulator.gamma
                
                reward = running_penalty + terminal_penalty
                total_reward += reward
            
            # Add negative reward as loss
            loss = -total_reward / self.batch_size
            total_loss += loss
        
        # Create tensor for backward pass
        loss_tensor = torch.tensor(total_loss, dtype=torch.float32, requires_grad=True, device=self.device)
        
        # Backward pass
        loss_tensor.backward()
        self.optimizer.step()
        
        return total_loss

#=============================================================================
# 3. Closed-Form Solution (Benchmark)
#=============================================================================

class ClosedFormController:
    """Closed-form controller based on PDE solution"""
    
    def __init__(self, simulator):
        """
        Initialize the controller
        
        Parameters:
        -----------
        simulator : OptimalExecutionSimulator
            Simulator for the optimal execution problem
        """
        self.simulator = simulator
        
        # Precompute h1 and h2 functions
        self._precompute_h_functions()
    
    def _precompute_h_functions(self):
        """Precompute h1 and h2 functions from ODEs"""
        # Parameters
        T = self.simulator.T
        dt = self.simulator.dt
        alpha = self.simulator.alpha
        kappa = self.simulator.kappa
        A = self.simulator.A
        phi = self.simulator.phi
        
        # Grid of time points
        t_grid = np.arange(T+1)
        
        # Initialize h1 and h2 at terminal time T
        h1 = np.zeros(T+1)
        h2 = np.zeros(T+1)
        h1[T] = 0
        h2[T] = -2 * A
        
        # Solve ODEs backwards in time
        for i in range(T-1, -1, -1):
            # Update h2 using Euler method
            h2_dot = -(2*phi - alpha**2/(2*kappa)) - (alpha/kappa)*h2[i+1] - (h2[i+1]**2)/(2*kappa)
            h2[i] = h2[i+1] - dt * h2_dot
            
            # Update h1 using Euler method
            h1_dot = -(alpha + h2[i+1])/(2*kappa) * h1[i+1]
            h1[i] = h1[i+1] - dt * h1_dot
        
        self.h1 = h1
        self.h2 = h2
    
    def control(self, state):
        """
        Return trading rate for a given state
        
        Parameters:
        -----------
        state : array
            State of the system (T-t, q)
            
        Returns:
        --------
        trading_rate : float
            Trading rate for the given state
        """
        remaining_time = state[0]
        inventory = state[1]
        
        # Get time index
        t = self.simulator.T - int(remaining_time)
        
        # Compute control
        alpha = self.simulator.alpha
        kappa = self.simulator.kappa
        
        trading_rate = self.h1[t]/(2*kappa) + (alpha + self.h2[t])/(2*kappa) * inventory
        
        return trading_rate

#=============================================================================
# 4. Model Explainability Functions
#=============================================================================

def project_controls_on_closed_form(controller, simulator, time_points):
    """
    Project controls from the neural network onto the closed-form manifold
    
    Parameters:
    -----------
    controller : NeuralNetworkController
        Controller to project
    simulator : OptimalExecutionSimulator
        Simulator for context
    time_points : list
        Time points to evaluate
        
    Returns:
    --------
    h1_tilde : array
        Projected h1 values
    h2_tilde : array
        Projected h2 values
    r_squared : array
        R-squared values for each time point
    """
    h1_tilde = np.zeros(len(time_points))
    h2_tilde = np.zeros(len(time_points))
    r_squared = np.zeros(len(time_points))
    
    # For each time point
    for i, t in enumerate(time_points):
        # Create range of inventory values
        inventory_values = np.linspace(-100, 0, 200)
        control_values = np.zeros_like(inventory_values)
        
        # Get control values for each inventory
        for j, q in enumerate(inventory_values):
            state = np.array([simulator.T - t, q])
            control_values[j] = controller.control(state)
        
        # Perform linear regression
        X = inventory_values.reshape(-1, 1)
        y = control_values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Extract coefficients
        intercept = model.intercept_
        slope = model.coef_[0]
        
        # Convert to h1 and h2
        h1_tilde[i] = 2 * simulator.kappa * intercept
        h2_tilde[i] = 2 * simulator.kappa * slope - simulator.alpha
        
        # Calculate R-squared
        r_squared[i] = model.score(X, y)
    
    return h1_tilde, h2_tilde, r_squared

#=============================================================================
# 5. Main Execution and Visualization
#=============================================================================

def main():
    """Main execution function"""
    # Create simulator
    simulator = OptimalExecutionSimulator(
        T=77,  # 77 five-minute bins in a trading day
        dt=1.0,
        initial_price=100.0,
        vol=0.1,
        alpha=0.1,
        kappa=0.1,
        A=0.01,
        phi=0.007,
        gamma=2.0,
        with_seasonality=False
    )
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create controllers
    closed_form = ClosedFormController(simulator)
    
    neural_net = NeuralNetworkController(
        simulator=simulator,
        multi_preference=False,
        hidden_dims=[5, 5, 5],
        learning_rate=5e-4,
        batch_size=64,
        device=device
    )
    
    multi_pref_net = NeuralNetworkController(
        simulator=simulator,
        multi_preference=True,
        hidden_dims=[5, 5, 5],
        learning_rate=5e-4,
        batch_size=64,
        device=device
    )
    
    # Train the neural network on simulated data
    print("Training neural network on simulated data...")
    history = neural_net.train_on_simulated_data(
        n_iterations=20000,  # Reduced for demonstration
        initial_inventory=-100.0
    )
    
    # Train the multi-preference neural network
    print("Training multi-preference neural network on simulated data...")
    multi_history = multi_pref_net.train_on_simulated_data(
        n_iterations=20000,  # Reduced for demonstration
        initial_inventory=-100.0
    )
    
    # Create simulator with seasonality
    simulator_seasonal = OptimalExecutionSimulator(
        T=77,
        dt=1.0,
        initial_price=100.0,
        vol=0.1,
        alpha=0.1,
        kappa=0.1,
        A=0.01,
        phi=0.007,
        gamma=2.0,
        with_seasonality=True
    )
    
    # Train on simulator with seasonality
    neural_net_seasonal = NeuralNetworkController(
        simulator=simulator_seasonal,
        multi_preference=False,
        hidden_dims=[5, 5, 5],
        learning_rate=5e-4,
        batch_size=64,
        device=device
    )
    
    print("Training neural network on simulated data with seasonality...")
    history_seasonal = neural_net_seasonal.train_on_simulated_data(
        n_iterations=20000,  # Reduced for demonstration
        initial_inventory=-100.0
    )
    
    # Create sub-diffusive simulator (gamma=1.5)
    simulator_subdiff = OptimalExecutionSimulator(
        T=77,
        dt=1.0,
        initial_price=100.0,
        vol=0.1,
        alpha=0.1,
        kappa=0.1,
        A=0.5,
        phi=0.1,
        gamma=1.5,
        with_seasonality=False
    )
    
    # Train on sub-diffusive simulator
    neural_net_subdiff = NeuralNetworkController(
        simulator=simulator_subdiff,
        multi_preference=False,
        hidden_dims=[5, 5, 5],
        learning_rate=5e-4,
        batch_size=64,
        device=device
    )
    
    print("Training neural network with sub-diffusive loss...")
    history_subdiff = neural_net_subdiff.train_on_simulated_data(
        n_iterations=20000,  # Reduced for demonstration
        initial_inventory=-100.0
    )
    
    # Evaluate and project each controller
    time_points = np.arange(simulator.T + 1)
    
    # Closed form values
    h1_closed = closed_form.h1
    h2_closed = closed_form.h2
    r_squared_closed = np.ones_like(time_points)
    
    # Project neural net controller
    h1_nn, h2_nn, r_squared_nn = project_controls_on_closed_form(
        neural_net, simulator, time_points
    )
    
    # Project multi-preference neural net controller
    h1_multi, h2_multi, r_squared_multi = project_controls_on_closed_form(
        multi_pref_net, simulator, time_points
    )
    
    # Project seasonal neural net controller
    h1_seasonal, h2_seasonal, r_squared_seasonal = project_controls_on_closed_form(
        neural_net_seasonal, simulator_seasonal, time_points
    )
    
    # Project sub-diffusive neural net controller
    h1_subdiff, h2_subdiff, r_squared_subdiff = project_controls_on_closed_form(
        neural_net_subdiff, simulator_subdiff, time_points
    )
    
    # Plot results
    plt.figure(figsize=(20, 15))
    
    # Plot h1 values
    plt.subplot(2, 2, 1)
    plt.plot(time_points, h1_closed, 'k-', label='Closed-form solution', linewidth=2)
    plt.plot(time_points, h1_nn, 'b-', label='NN on simulations', linewidth=2)
    plt.plot(time_points, h1_seasonal, 'g--', label='NN with seasonality', linewidth=2)
    plt.plot(time_points, h1_multi, 'r-.', label='Multi-preference NN', linewidth=2)
    plt.plot(time_points, h1_subdiff, 'm:', label='NN with γ=3/2', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('h1(t)')
    plt.legend()
    plt.grid(True)
    
    # Plot h2 values
    plt.subplot(2, 2, 2)
    plt.plot(time_points, h2_closed, 'k-', label='Closed-form solution', linewidth=2)
    plt.plot(time_points, h2_nn, 'b-', label='NN on simulations', linewidth=2)
    plt.plot(time_points, h2_seasonal, 'g--', label='NN with seasonality', linewidth=2)
    plt.plot(time_points, h2_multi, 'r-.', label='Multi-preference NN', linewidth=2)
    plt.plot(time_points, h2_subdiff, 'm:', label='NN with γ=3/2', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('h2(t)')
    plt.legend()
    plt.grid(True)
    
    # Plot R-squared values
    plt.subplot(2, 2, 3)
    plt.plot(time_points, r_squared_closed, 'k-', label='Closed-form solution', linewidth=2)
    plt.plot(time_points, r_squared_nn, 'b-', label='NN on simulations', linewidth=2)
    plt.plot(time_points, r_squared_seasonal, 'g--', label='NN with seasonality', linewidth=2)
    plt.plot(time_points, r_squared_multi, 'r-.', label='Multi-preference NN', linewidth=2)
    plt.plot(time_points, r_squared_subdiff, 'm:', label='NN with γ=3/2', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('R²(t)')
    plt.legend()
    plt.grid(True)
    
    # Plot average control
    plt.subplot(2, 2, 4)
    
    # Simulate trajectories
    inventory = -100.0
    
    # Closed-form trajectory
    cf_traj = simulator.simulate_trajectory(closed_form.control, inventory)
    nn_traj = simulator.simulate_trajectory(neural_net.control, inventory)
    multi_traj = simulator.simulate_trajectory(multi_pref_net.control, inventory)
    seasonal_traj = simulator_seasonal.simulate_trajectory(neural_net_seasonal.control, inventory)
    subdiff_traj = simulator_subdiff.simulate_trajectory(neural_net_subdiff.control, inventory)
    
    plt.plot(cf_traj['time_path'][:-1], cf_traj['trading_rates'], 'k-', label='Closed-form solution', linewidth=2)
    plt.plot(nn_traj['time_path'][:-1], nn_traj['trading_rates'], 'b-', label='NN on simulations', linewidth=2)
    plt.plot(seasonal_traj['time_path'][:-1], seasonal_traj['trading_rates'], 'g--', label='NN with seasonality', linewidth=2)
    plt.plot(multi_traj['time_path'][:-1], multi_traj['trading_rates'], 'r-.', label='Multi-preference NN', linewidth=2)
    plt.plot(subdiff_traj['time_path'][:-1], subdiff_traj['trading_rates'], 'm:', label='NN with γ=3/2', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Average Control Process (ν)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('trading_controls_comparison.png')
    plt.show()
    
    # Plot inventory paths
    plt.figure(figsize=(12, 8))
    plt.plot(cf_traj['time_path'], cf_traj['inventory_path'], 'k-', label='Closed-form solution', linewidth=2)
    plt.plot(nn_traj['time_path'], nn_traj['inventory_path'], 'b-', label='NN on simulations', linewidth=2)
    plt.plot(seasonal_traj['time_path'], seasonal_traj['inventory_path'], 'g--', label='NN with seasonality', linewidth=2)
    plt.plot(multi_traj['time_path'], multi_traj['inventory_path'], 'r-.', label='Multi-preference NN', linewidth=2)
    plt.plot(subdiff_traj['time_path'], subdiff_traj['inventory_path'], 'm:', label='NN with γ=3/2', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Inventory')
    plt.legend()
    plt.grid(True)
    plt.savefig('inventory_paths.png')
    plt.show()
    
    # Print final metrics
    print("\nFinal Results:")
    print(f"Closed-form final inventory: {cf_traj['inventory_path'][-1]:.2f}")
    print(f"Neural Net final inventory: {nn_traj['inventory_path'][-1]:.2f}")
    print(f"Multi-pref Net final inventory: {multi_traj['inventory_path'][-1]:.2f}")
    print(f"Seasonal Net final inventory: {seasonal_traj['inventory_path'][-1]:.2f}")
    print(f"Sub-diffusive Net final inventory: {subdiff_traj['inventory_path'][-1]:.2f}")
    
    print(f"\nClosed-form total reward: {cf_traj['total_reward']:.2f}")
    print(f"Neural Net total reward: {nn_traj['total_reward']:.2f}")
    print(f"Multi-pref Net total reward: {multi_traj['total_reward']:.2f}")
    print(f"Seasonal Net total reward: {seasonal_traj['total_reward']:.2f}")
    print(f"Sub-diffusive Net total reward: {subdiff_traj['total_reward']:.2f}")

if __name__ == "__main__":
    main()