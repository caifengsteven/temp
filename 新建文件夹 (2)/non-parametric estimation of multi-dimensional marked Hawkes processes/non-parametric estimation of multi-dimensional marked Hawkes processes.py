import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from scipy.stats import lognorm
import time
import seaborn as sns
from sklearn.mixture import GaussianMixture

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class MarkedHawkesSimulator:
    """
    Simulator for a marked Hawkes process using the thinning algorithm
    """
    def __init__(self, mu, kernel_func, mark_dist, end_time):
        """
        Initialize the simulator
        
        Parameters:
        -----------
        mu : float
            Base intensity
        kernel_func : function
            Kernel function that takes time difference and mark as input
        mark_dist : function
            Function to sample marks
        end_time : float
            End time for simulation
        """
        self.mu = mu
        self.kernel_func = kernel_func
        self.mark_dist = mark_dist
        self.end_time = end_time
        
    def simulate(self):
        """
        Simulate a marked Hawkes process
        
        Returns:
        --------
        event_times : list
            List of event times
        marks : list
            List of marks associated with each event
        """
        t = 0
        event_times = []
        marks = []
        
        while t < self.end_time:
            # Calculate current intensity
            history_effect = 0
            for i in range(len(event_times)):
                history_effect += self.kernel_func(t - event_times[i], marks[i])
            
            current_intensity = max(self.mu + history_effect, 0)
            
            # Generate next candidate event time
            u = np.random.uniform(0, 1)
            dt = -np.log(u) / current_intensity if current_intensity > 0 else float('inf')
            t_next = t + dt
            
            if t_next > self.end_time:
                break
                
            # Accept/reject the candidate
            u = np.random.uniform(0, 1)
            
            # Recalculate intensity at the new time
            history_effect = 0
            for i in range(len(event_times)):
                history_effect += self.kernel_func(t_next - event_times[i], marks[i])
            
            new_intensity = max(self.mu + history_effect, 0)
            
            if u <= new_intensity / current_intensity:
                t = t_next
                event_times.append(t)
                mark = self.mark_dist()
                marks.append(mark)
            else:
                t = t_next
        
        return np.array(event_times), np.array(marks)

class SNHWithMarks(keras.Model):
    """
    Shallow Neural Hawkes with marks model
    """
    def __init__(self, num_neurons=64, base_intensity=0.7):
        super(SNHWithMarks, self).__init__()
        self.num_neurons = num_neurons
        self.base_intensity = tf.Variable(base_intensity, dtype=tf.float32)
        
        # Initialize network parameters
        self.a1 = tf.Variable(tf.random.uniform([num_neurons], -0.5, 0.5), dtype=tf.float32)
        self.a2 = tf.Variable(tf.random.uniform([num_neurons], -0.2, 0.2), dtype=tf.float32)
        self.a3 = tf.Variable(tf.random.uniform([num_neurons], -1.0, 0.0), dtype=tf.float32)
        self.b1 = tf.Variable(tf.random.uniform([num_neurons], 0.0, 0.03), dtype=tf.float32)
        self.b2 = tf.Variable(tf.random.uniform([1], -0.1, 0.0), dtype=tf.float32)
    
    def kernel(self, dt, mark):
        """
        Compute the kernel function for given time difference and mark
        
        Parameters:
        -----------
        dt : tensor
            Time difference
        mark : tensor
            Mark value
            
        Returns:
        --------
        kernel_value : tensor
            Value of the kernel function
        """
        inner = tf.maximum(self.a1 * dt + self.a2 * mark + self.b1, 0)
        output = tf.exp(self.b2 + tf.reduce_sum(self.a3 * inner))
        return output
    
    def compute_intensity(self, t, history_times, history_marks):
        """
        Compute the intensity at time t given history
        
        Parameters:
        -----------
        t : float
            Current time
        history_times : array
            Times of past events
        history_marks : array
            Marks of past events
            
        Returns:
        --------
        intensity : float
            Intensity at time t
        """
        # Filter history up to time t
        mask = history_times < t
        relevant_times = history_times[mask]
        relevant_marks = history_marks[mask]
        
        # Base intensity
        intensity = self.base_intensity
        
        # Add effect of past events
        if len(relevant_times) > 0:
            dt = t - relevant_times
            for i in range(len(dt)):
                intensity += self.kernel(dt[i], relevant_marks[i])
        
        return intensity
    
    def log_likelihood(self, event_times, marks):
        """
        Compute the log-likelihood of the model
        
        Parameters:
        -----------
        event_times : array
            Array of event times
        marks : array
            Array of marks
            
        Returns:
        --------
        log_likelihood : float
            Log-likelihood value
        """
        log_lik = 0.0
        
        # Sort events by time
        indices = np.argsort(event_times)
        sorted_times = event_times[indices]
        sorted_marks = marks[indices]
        
        for i in range(len(sorted_times)):
            t = sorted_times[i]
            
            # Log intensity term
            intensity = self.compute_intensity(t, sorted_times[:i], sorted_marks[:i])
            log_lik += tf.math.log(intensity)
            
            # Integrated intensity term (approximation)
            if i > 0:
                prev_t = sorted_times[i-1]
                dt = t - prev_t
                # Simple approximation with midpoint rule
                mid_t = prev_t + dt/2
                mid_intensity = self.compute_intensity(mid_t, sorted_times[:i], sorted_marks[:i])
                log_lik -= mid_intensity * dt
        
        # Subtract intensity for the last interval
        if len(sorted_times) > 0:
            last_t = sorted_times[-1]
            end_t = last_t + 1.0  # Arbitrary extension
            dt = end_t - last_t
            mid_t = last_t + dt/2
            mid_intensity = self.compute_intensity(mid_t, sorted_times, sorted_marks)
            log_lik -= mid_intensity * dt
            
        return log_lik
    
    def train_step(self, data):
        """
        Perform a training step
        
        Parameters:
        -----------
        data : tuple
            Tuple of (event_times, marks)
            
        Returns:
        --------
        loss : float
            Negative log-likelihood
        """
        event_times, marks = data
        
        with tf.GradientTape() as tape:
            neg_log_lik = -self.log_likelihood(event_times, marks)
        
        # Get gradients
        gradients = tape.gradient(neg_log_lik, self.trainable_variables)
        
        # Apply gradients with different learning rates
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {"loss": neg_log_lik}

# Define the kernel function for simulation
def true_kernel(dt, mark):
    """
    True kernel function: m * exp(-t * (1 + 5*m))
    """
    return mark * np.exp(-dt * (1 + 5 * mark))

# Define mark distribution
def sample_mark():
    """
    Sample a mark from lognormal distribution
    """
    return lognorm.rvs(s=0.5, scale=1.0)

# Simulate data
simulator = MarkedHawkesSimulator(
    mu=0.7,
    kernel_func=true_kernel,
    mark_dist=sample_mark,
    end_time=2000
)

print("Simulating marked Hawkes process...")
event_times, marks = simulator.simulate()
print(f"Simulated {len(event_times)} events")

# Data preprocessing
def preprocess_data(event_times, marks):
    """
    Preprocess data by scaling
    """
    t_max = np.max(event_times)
    n = len(event_times)
    
    # Scale times and marks
    scaled_times = event_times * n / t_max
    scaled_marks = marks / np.mean(marks)
    
    return scaled_times, scaled_marks

scaled_times, scaled_marks = preprocess_data(event_times, marks)

# Split data into train, validation, and test sets
def split_data(times, marks, train_ratio=0.7, val_ratio=0.15):
    """
    Split data into train, validation, and test sets
    """
    n = len(times)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_times = times[:train_size]
    train_marks = marks[:train_size]
    
    val_times = times[train_size:train_size+val_size]
    val_marks = marks[train_size:train_size+val_size]
    
    test_times = times[train_size+val_size:]
    test_marks = marks[train_size+val_size:]
    
    return (train_times, train_marks), (val_times, val_marks), (test_times, test_marks)

(train_times, train_marks), (val_times, val_marks), (test_times, test_marks) = split_data(scaled_times, scaled_marks)

# Train the model
model = SNHWithMarks()

# Define optimizers with different learning rates
hidden_layer_vars = [model.a1, model.a2, model.b1]
output_layer_vars = [model.a3, model.b2]
base_intensity_var = [model.base_intensity]

# Use custom training loop with different learning rates
print("Training model...")
epochs = 30
batch_size = 100
patience = 10
best_val_loss = float('inf')
patience_counter = 0

train_losses = []
val_losses = []

# Optimizers with different learning rates
optimizer_hidden = keras.optimizers.Adam(learning_rate=2e-3)
optimizer_output = keras.optimizers.Adam(learning_rate=2e-2)
optimizer_base = keras.optimizers.Adam(learning_rate=1e-3)

start_time = time.time()

for epoch in range(epochs):
    # Shuffle data
    indices = np.random.permutation(len(train_times))
    shuffled_times = train_times[indices]
    shuffled_marks = train_marks[indices]
    
    # Training
    epoch_loss = 0
    n_batches = 0
    
    for i in range(0, len(shuffled_times), batch_size):
        batch_times = shuffled_times[i:i+batch_size]
        batch_marks = shuffled_marks[i:i+batch_size]
        
        with tf.GradientTape() as tape:
            neg_log_lik = -model.log_likelihood(batch_times, batch_marks)
        
        # Get gradients
        gradients = tape.gradient(neg_log_lik, model.trainable_variables)
        
        # Apply gradients with different learning rates based on parameter groups
        hidden_grads = [gradients[i] for i in range(len(gradients)) if model.trainable_variables[i] in hidden_layer_vars]
        output_grads = [gradients[i] for i in range(len(gradients)) if model.trainable_variables[i] in output_layer_vars]
        base_grads = [gradients[i] for i in range(len(gradients)) if model.trainable_variables[i] in base_intensity_var]
        
        optimizer_hidden.apply_gradients(zip(hidden_grads, hidden_layer_vars))
        optimizer_output.apply_gradients(zip(output_grads, output_layer_vars))
        optimizer_base.apply_gradients(zip(base_grads, base_intensity_var))
        
        epoch_loss += neg_log_lik.numpy()
        n_batches += 1
    
    train_loss = epoch_loss / n_batches
    train_losses.append(train_loss)
    
    # Validation
    val_loss = -model.log_likelihood(val_times, val_marks).numpy()
    val_losses.append(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")

# Evaluate on test set
test_loss = -model.log_likelihood(test_times, test_marks).numpy()
print(f"Test Loss: {test_loss:.4f}")

# GMM for marks
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(marks.reshape(-1, 1))

# Plot the estimated kernel
def plot_kernels():
    """
    Plot the true and estimated kernels
    """
    t_range = np.linspace(0, 5, 100)
    m_range = np.linspace(0.5, 3, 100)
    T, M = np.meshgrid(t_range, m_range)
    
    # True kernel
    Z_true = np.zeros_like(T)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            Z_true[i, j] = true_kernel(T[i, j], M[i, j])
    
    # Estimated kernel
    Z_est = np.zeros_like(T)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            Z_est[i, j] = model.kernel(tf.constant(T[i, j], dtype=tf.float32), 
                                      tf.constant(M[i, j], dtype=tf.float32)).numpy()
    
    # Error
    Z_error = np.abs(Z_true - Z_est)
    
    # Create figure with 3D plots
    fig = plt.figure(figsize=(18, 6))
    
    # True kernel
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(T, M, Z_true, cmap='viridis')
    ax1.set_xlabel('Time difference (t)')
    ax1.set_ylabel('Mark (m)')
    ax1.set_zlabel('Kernel value')
    ax1.set_title('True Kernel')
    
    # Estimated kernel
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(T, M, Z_est, cmap='viridis')
    ax2.set_xlabel('Time difference (t)')
    ax2.set_ylabel('Mark (m)')
    ax2.set_zlabel('Kernel value')
    ax2.set_title('Estimated Kernel')
    
    # Error
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(T, M, Z_error, cmap='viridis')
    ax3.set_xlabel('Time difference (t)')
    ax3.set_ylabel('Mark (m)')
    ax3.set_zlabel('Error')
    ax3.set_title('Absolute Error')
    
    plt.tight_layout()
    plt.show()

# Plot QQ plot for interarrival times
def plot_qq():
    """
    Plot QQ plot for interarrival times
    """
    # Calculate empirical interarrival times
    diff_times = np.diff(test_times)
    
    # Calculate predicted cumulative intensity for each interarrival time
    predicted_intensities = []
    for i in range(len(test_times)-1):
        t_start = test_times[i]
        t_end = test_times[i+1]
        history_times = np.concatenate([train_times, val_times, test_times[:i+1]])
        history_marks = np.concatenate([train_marks, val_marks, test_marks[:i+1]])
        
        # Simple approximation of the integrated intensity
        n_points = 10
        t_points = np.linspace(t_start, t_end, n_points)
        intensity_sum = 0
        for t in t_points:
            intensity_sum += model.compute_intensity(t, history_times, history_marks).numpy()
        
        integrated_intensity = intensity_sum * (t_end - t_start) / n_points
        predicted_intensities.append(integrated_intensity)
    
    # Calculate theoretical quantiles (exponential with rate 1)
    theoretical_quantiles = -np.log(1 - np.linspace(0.01, 0.99, 99))
    
    # Calculate empirical quantiles from predicted intensities
    empirical_quantiles = np.quantile(predicted_intensities, np.linspace(0.01, 0.99, 99))
    
    # Plot QQ plot
    plt.figure(figsize=(8, 8))
    plt.scatter(theoretical_quantiles, empirical_quantiles)
    plt.plot([0, max(theoretical_quantiles)], [0, max(theoretical_quantiles)], 'r--')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Empirical Quantiles')
    plt.title('QQ Plot for Interarrival Times')
    plt.grid(True)
    plt.show()

# Plot the estimated kernels
plot_kernels()

# Plot QQ plot
plot_qq()

# Plot mark distribution and GMM fit
plt.figure(figsize=(10, 6))
sns.histplot(marks, bins=30, kde=True, stat='density', label='Empirical')

# Plot GMM density
x = np.linspace(min(marks), max(marks), 1000)
gmm_density = np.exp(gmm.score_samples(x.reshape(-1, 1)))
plt.plot(x, gmm_density, 'r-', label='GMM Fit')

plt.title('Mark Distribution and GMM Fit')
plt.xlabel('Mark Value')
plt.ylabel('Density')
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Negative Log-Likelihood')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()
