import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Simulating data based on the paper's description
# Time period: quarterly data from 1960 to 2020 (approximately 240 quarters)
periods = 240
time_index = pd.date_range(start='1960-01-01', periods=periods, freq='Q')

# Generate simulated data
def generate_data():
    # Target inflation rate and equilibrium real interest rate
    pi_star = 2.0
    r_star = 2.0
    
    # Generate inflation rate (with some randomness and trend)
    inflation = np.zeros(periods)
    inflation[0] = 2.0  # Starting inflation
    
    # Add some realism to inflation dynamics
    for t in range(1, periods):
        if t < 60:  # 1960s-1970s: rising inflation
            drift = 0.1
        elif t < 100:  # Late 1970s-early 1980s: high inflation
            drift = 0
        elif t < 160:  # 1980s-1990s: falling inflation
            drift = -0.05
        else:  # 2000s-2020: stable low inflation
            drift = 0
        
        # Random walk with drift for inflation
        inflation[t] = inflation[t-1] + drift + np.random.normal(0, 0.5)
    
    # Ensure inflation stays somewhat realistic
    inflation = np.clip(inflation, 0, 14)
    
    # Generate output gap (business cycle dynamics)
    output_gap = np.zeros(periods)
    # Create business cycles of roughly 5-7 years
    for t in range(periods):
        # Combining multiple sine waves for realistic business cycles
        cycle1 = 3 * np.sin(2 * np.pi * t / 20)  # ~5 year cycle
        cycle2 = 1.5 * np.sin(2 * np.pi * t / 28)  # ~7 year cycle
        cycle3 = 0.5 * np.sin(2 * np.pi * t / 12)  # ~3 year cycle
        output_gap[t] = cycle1 + cycle2 + cycle3 + np.random.normal(0, 0.5)
    
    # Add some recessions (stronger negative output gaps)
    recession_periods = [30, 80, 120, 160, 200]  # Approximate recession periods
    for rp in recession_periods:
        if rp < periods:
            output_gap[rp:rp+8] -= 2.0  # Deeper recession
    
    # Calculate federal funds rate using the original Taylor Rule
    taylor_rate = r_star + inflation + 0.5 * (inflation - pi_star) + 0.5 * output_gap
    
    # Add some deviations for actual federal funds rate (to simulate Fed behavior)
    actual_rate = taylor_rate + np.random.normal(0, 1.5, size=periods)
    
    # Add specific behavior for bubble bursts (as mentioned in the paper)
    bubble_bursts = [96, 168, 196]  # Approximating the 1984, 2001, 2009 recessions
    for bb in bubble_bursts:
        if bb < periods:
            actual_rate[bb:bb+8] -= 3.0  # Stronger rate cuts during bubble bursts
    
    # Ensure rates don't go extremely negative (Fed's behavior)
    actual_rate = np.maximum(actual_rate, 0.25)
    
    return pd.DataFrame({
        'date': time_index,
        'inflation': inflation,
        'output_gap': output_gap,
        'taylor_rate': taylor_rate,
        'actual_rate': actual_rate
    })

# Generate the data
data = generate_data()

# Visualize the simulated data
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(data['date'], data['inflation'], label='Inflation Rate')
plt.axhline(y=2.0, color='r', linestyle='--', label='Target Inflation (2%)')
plt.title('Simulated Inflation Rate (1960-2020)')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(data['date'], data['output_gap'], label='Output Gap')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Simulated Output Gap (1960-2020)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(data['date'], data['actual_rate'], label='Actual Federal Funds Rate')
plt.plot(data['date'], data['taylor_rate'], label='Taylor Rule Estimate')
plt.title('Federal Funds Rate vs Taylor Rule Estimate')
plt.legend()

plt.tight_layout()
plt.savefig('simulated_data.png')
plt.close()

# 1. Original Taylor Rule
# The formula is: i_t = π_t + r* + 0.5(π_t - π*) + 0.5(y_t - y*_t)
# Where:
# i_t = federal funds rate
# π_t = inflation rate
# r* = equilibrium real interest rate (set at 2%)
# π* = target inflation rate (set at 2%)
# y_t - y*_t = output gap

r_star = 2.0
pi_star = 2.0

def original_taylor_rule(inflation, output_gap):
    return inflation + r_star + 0.5 * (inflation - pi_star) + 0.5 * output_gap

# Apply the original Taylor Rule
data['taylor_original'] = original_taylor_rule(data['inflation'], data['output_gap'])

# 2. OLS Regression Method
# As described in the paper, we need to adjust for multicollinearity
# We're estimating: i_t = α + θ_π * π_t + β_y * (y_t - y*_t) + ε_t
# Where θ_π = β_1 + β_π

X = pd.DataFrame({
    'inflation': data['inflation'],
    'output_gap': data['output_gap']
})
X = sm.add_constant(X)
y = data['actual_rate']

model = sm.OLS(y, X).fit()
print("OLS Regression Results:")
print(model.summary())

# Extract coefficients
alpha = model.params[0]  # Intercept
theta_pi = model.params[1]  # Coefficient for inflation
beta_y = model.params[2]  # Coefficient for output gap

# Calculate β_π and β_1
# From the paper: α = r* - β_π * π*
# Therefore: β_π = (r* - α) / π*
beta_pi = (r_star - alpha) / pi_star
beta_1 = theta_pi - beta_pi

print(f"\nCalculated Coefficients for OLS Taylor Rule:")
print(f"β_1 (coefficient for standalone inflation): {beta_1:.3f}")
print(f"β_π (coefficient for inflation gap): {beta_pi:.3f}")
print(f"β_y (coefficient for output gap): {beta_y:.3f}")

# Apply the OLS Taylor Rule
def ols_taylor_rule(inflation, output_gap):
    return r_star + beta_1 * inflation + beta_pi * (inflation - pi_star) + beta_y * output_gap

data['taylor_ols'] = ols_taylor_rule(data['inflation'], data['output_gap'])

# 3. Machine Learning Method using PyTorch
# Prepare data for ML
X_ml = data[['inflation', 'output_gap']].values
y_ml = data['actual_rate'].values

# Scale the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_ml_scaled = scaler_X.fit_transform(X_ml)
y_ml_scaled = scaler_y.fit_transform(y_ml.reshape(-1, 1)).flatten()

# Convert data to PyTorch tensors
X_tensor = torch.FloatTensor(X_ml_scaled)
y_tensor = torch.FloatTensor(y_ml_scaled)

# Create a dataset and dataloader
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_size)

# Define the neural network model with one hidden layer (as in the paper)
class TaylorNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=6, output_size=1):
        super(TaylorNN, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.output(x)
        return x

# Create the model instance
ml_model = TaylorNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(ml_model.parameters(), lr=0.01)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=1000):
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    return train_losses, val_losses

# Train the model
train_losses, val_losses = train_model(ml_model, train_loader, val_loader, criterion, optimizer)

# Visualize training progress
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Neural Network Training Progress')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('ml_training_pytorch.png')
plt.close()

# Predict using the ML model
ml_model.eval()
with torch.no_grad():
    y_ml_pred_scaled = ml_model(X_tensor).numpy().flatten()

y_ml_pred = scaler_y.inverse_transform(y_ml_pred_scaled.reshape(-1, 1)).flatten()
data['taylor_ml'] = y_ml_pred

# Compare the three methods
plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
plt.plot(data['date'], data['actual_rate'], 'k-', label='Actual Federal Funds Rate')
plt.plot(data['date'], data['taylor_original'], 'r--', label='Original Taylor Rule')
plt.plot(data['date'], data['taylor_ols'], 'g-.', label='OLS Taylor Rule')
plt.plot(data['date'], data['taylor_ml'], 'b:', label='ML Taylor Rule')
plt.title('Comparison of Taylor Rule Methods')
plt.legend()

# Plot the residuals
plt.subplot(2, 1, 2)
plt.plot(data['date'], data['actual_rate'] - data['taylor_original'], 'r--', label='Original Taylor Rule Residuals')
plt.plot(data['date'], data['actual_rate'] - data['taylor_ols'], 'g-.', label='OLS Taylor Rule Residuals')
plt.plot(data['date'], data['actual_rate'] - data['taylor_ml'], 'b:', label='ML Taylor Rule Residuals')
plt.axhline(y=0, color='k', linestyle='-')
plt.title('Residuals (Actual - Estimated)')
plt.legend()

plt.tight_layout()
plt.savefig('taylor_rule_comparison_pytorch.png')
plt.close()

# Calculate and compare error metrics
print("\nError Metrics:")
print("Original Taylor Rule:")
print(f"  Mean Absolute Error: {mean_absolute_error(data['actual_rate'], data['taylor_original']):.4f}")
print(f"  Root Mean Squared Error: {np.sqrt(mean_squared_error(data['actual_rate'], data['taylor_original'])):.4f}")
print(f"  Sum of Residuals: {np.sum(data['actual_rate'] - data['taylor_original']):.4f}")
print(f"  Sum of Absolute Residuals: {np.sum(np.abs(data['actual_rate'] - data['taylor_original'])):.4f}")

print("\nOLS Taylor Rule:")
print(f"  Mean Absolute Error: {mean_absolute_error(data['actual_rate'], data['taylor_ols']):.4f}")
print(f"  Root Mean Squared Error: {np.sqrt(mean_squared_error(data['actual_rate'], data['taylor_ols'])):.4f}")
print(f"  Sum of Residuals: {np.sum(data['actual_rate'] - data['taylor_ols']):.4f}")
print(f"  Sum of Absolute Residuals: {np.sum(np.abs(data['actual_rate'] - data['taylor_ols'])):.4f}")

print("\nML Taylor Rule (PyTorch):")
print(f"  Mean Absolute Error: {mean_absolute_error(data['actual_rate'], data['taylor_ml']):.4f}")
print(f"  Root Mean Squared Error: {np.sqrt(mean_squared_error(data['actual_rate'], data['taylor_ml'])):.4f}")
print(f"  Sum of Residuals: {np.sum(data['actual_rate'] - data['taylor_ml']):.4f}")
print(f"  Sum of Absolute Residuals: {np.sum(np.abs(data['actual_rate'] - data['taylor_ml'])):.4f}")

# Create a scatter plot to visualize the relationship between actual and estimated rates
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(data['actual_rate'], data['taylor_original'], alpha=0.6)
plt.plot([0, 20], [0, 20], 'k--')  # Perfect prediction line
plt.title('Original Taylor Rule vs Actual')
plt.xlabel('Actual Federal Funds Rate')
plt.ylabel('Estimated Rate')

plt.subplot(1, 3, 2)
plt.scatter(data['actual_rate'], data['taylor_ols'], alpha=0.6)
plt.plot([0, 20], [0, 20], 'k--')  # Perfect prediction line
plt.title('OLS Taylor Rule vs Actual')
plt.xlabel('Actual Federal Funds Rate')
plt.ylabel('Estimated Rate')

plt.subplot(1, 3, 3)
plt.scatter(data['actual_rate'], data['taylor_ml'], alpha=0.6)
plt.plot([0, 20], [0, 20], 'k--')  # Perfect prediction line
plt.title('ML Taylor Rule (PyTorch) vs Actual')
plt.xlabel('Actual Federal Funds Rate')
plt.ylabel('Estimated Rate')

plt.tight_layout()
plt.savefig('accuracy_comparison_pytorch.png')
plt.close()

# Analyze bubble burst periods
bubble_periods = [
    (96, 104),   # Approximating the 1984 recession (Energy Crisis)
    (168, 176),  # Approximating the 2001 recession (Dot-Com Bubble)
    (196, 204)   # Approximating the 2009 recession (Housing Bubble)
]

# Create a "bubble burst" dummy variable
data['bubble_burst'] = 0
for start, end in bubble_periods:
    if start < len(data):
        end = min(end, len(data))
        data.loc[start:end, 'bubble_burst'] = 1

# Plot the performance during bubble burst periods
plt.figure(figsize=(15, 12))
for i, (start, end) in enumerate(bubble_periods):
    if start >= len(data):
        continue
        
    end = min(end, len(data))
    period_data = data.iloc[max(0, start-4):min(len(data), end+4)]
    
    plt.subplot(3, 1, i+1)
    plt.plot(period_data['date'], period_data['actual_rate'], 'k-', label='Actual Federal Funds Rate')
    plt.plot(period_data['date'], period_data['taylor_original'], 'r--', label='Original Taylor Rule')
    plt.plot(period_data['date'], period_data['taylor_ols'], 'g-.', label='OLS Taylor Rule')
    plt.plot(period_data['date'], period_data['taylor_ml'], 'b:', label='ML Taylor Rule')
    
    # Highlight the bubble burst period
    bubble_start = period_data['date'].iloc[4]  # Adjust for the offset
    bubble_end = period_data['date'].iloc[min(12, len(period_data)-1)]  # Adjust for the offset
    plt.axvspan(bubble_start, bubble_end, alpha=0.2, color='gray')
    
    recession_names = ["1984 Energy Crisis", "2001 Dot-Com Bubble", "2009 Housing Bubble"]
    plt.title(f'Performance during {recession_names[i]}')
    plt.legend()

plt.tight_layout()
plt.savefig('bubble_periods_comparison_pytorch.png')
plt.close()

# Extra: Test an improved ML model with bubble burst indicator
# Create a new feature set with the bubble burst indicator
X_ml_improved = np.column_stack([X_ml, data['bubble_burst'].values])
X_ml_improved_scaled = StandardScaler().fit_transform(X_ml_improved)

# Convert to PyTorch tensors
X_improved_tensor = torch.FloatTensor(X_ml_improved_scaled)

# Create improved dataset and dataloader
improved_dataset = TensorDataset(X_improved_tensor, y_tensor)
train_size = int(0.8 * len(improved_dataset))
val_size = len(improved_dataset) - train_size
improved_train_dataset, improved_val_dataset = torch.utils.data.random_split(improved_dataset, [train_size, val_size])
improved_train_loader = DataLoader(improved_train_dataset, batch_size=32, shuffle=True)
improved_val_loader = DataLoader(improved_val_dataset, batch_size=val_size)

# Define an improved neural network model
class ImprovedTaylorNN(nn.Module):
    def __init__(self, input_size=3, hidden_size1=8, hidden_size2=4, output_size=1):
        super(ImprovedTaylorNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.sigmoid1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.sigmoid2 = nn.Sigmoid()
        self.output = nn.Linear(hidden_size2, output_size)
        
    def forward(self, x):
        x = self.sigmoid1(self.fc1(x))
        x = self.sigmoid2(self.fc2(x))
        x = self.output(x)
        return x

# Create and train the improved model
ml_model_improved = ImprovedTaylorNN()
optimizer_improved = optim.Adam(ml_model_improved.parameters(), lr=0.01)

train_losses_improved, val_losses_improved = train_model(
    ml_model_improved, 
    improved_train_loader, 
    improved_val_loader, 
    criterion, 
    optimizer_improved
)

# Visualize training progress for improved model
plt.figure(figsize=(10, 6))
plt.plot(train_losses_improved, label='Training Loss')
plt.plot(val_losses_improved, label='Validation Loss')
plt.title('Improved Neural Network Training Progress')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('ml_improved_training_pytorch.png')
plt.close()

# Predict using the improved ML model
ml_model_improved.eval()
with torch.no_grad():
    y_ml_improved_pred_scaled = ml_model_improved(X_improved_tensor).numpy().flatten()

y_ml_improved_pred = scaler_y.inverse_transform(y_ml_improved_pred_scaled.reshape(-1, 1)).flatten()
data['taylor_ml_improved'] = y_ml_improved_pred

# Compare the original ML model with the improved one
plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
plt.plot(data['date'], data['actual_rate'], 'k-', label='Actual Federal Funds Rate')
plt.plot(data['date'], data['taylor_ml'], 'b:', label='ML Taylor Rule')
plt.plot(data['date'], data['taylor_ml_improved'], 'm-', label='ML with Bubble Indicator')
plt.title('Comparison of ML Models (PyTorch)')
plt.legend()

# Plot the residuals
plt.subplot(2, 1, 2)
plt.plot(data['date'], data['actual_rate'] - data['taylor_ml'], 'b:', label='ML Taylor Rule Residuals')
plt.plot(data['date'], data['actual_rate'] - data['taylor_ml_improved'], 'm-', label='ML with Bubble Indicator Residuals')
plt.axhline(y=0, color='k', linestyle='-')
plt.title('Residuals (Actual - Estimated)')
plt.legend()

plt.tight_layout()
plt.savefig('ml_improved_comparison_pytorch.png')
plt.close()

# Calculate and compare error metrics for the improved model
print("\nImproved ML Taylor Rule (with Bubble Indicator, PyTorch):")
print(f"  Mean Absolute Error: {mean_absolute_error(data['actual_rate'], data['taylor_ml_improved']):.4f}")
print(f"  Root Mean Squared Error: {np.sqrt(mean_squared_error(data['actual_rate'], data['taylor_ml_improved'])):.4f}")
print(f"  Sum of Residuals: {np.sum(data['actual_rate'] - data['taylor_ml_improved']):.4f}")
print(f"  Sum of Absolute Residuals: {np.sum(np.abs(data['actual_rate'] - data['taylor_ml_improved'])):.4f}")

# Plot the performance during bubble burst periods with the improved model
plt.figure(figsize=(15, 12))
for i, (start, end) in enumerate(bubble_periods):
    if start >= len(data):
        continue
        
    end = min(end, len(data))
    period_data = data.iloc[max(0, start-4):min(len(data), end+4)]
    
    plt.subplot(3, 1, i+1)
    plt.plot(period_data['date'], period_data['actual_rate'], 'k-', label='Actual Federal Funds Rate')
    plt.plot(period_data['date'], period_data['taylor_ml'], 'b:', label='ML Taylor Rule')
    plt.plot(period_data['date'], period_data['taylor_ml_improved'], 'm-', label='ML with Bubble Indicator')
    
    # Highlight the bubble burst period
    bubble_start = period_data['date'].iloc[4]  # Adjust for the offset
    bubble_end = period_data['date'].iloc[min(12, len(period_data)-1)]  # Adjust for the offset
    plt.axvspan(bubble_start, bubble_end, alpha=0.2, color='gray')
    
    recession_names = ["1984 Energy Crisis", "2001 Dot-Com Bubble", "2009 Housing Bubble"]
    plt.title(f'Performance during {recession_names[i]}')
    plt.legend()

plt.tight_layout()
plt.savefig('bubble_periods_improved_comparison_pytorch.png')
plt.close()

# Custom visualization to show the nonlinear relationship captured by the ML model
# Create a grid of inflation and output gap values
inflation_range = np.linspace(-2, 14, 50)
output_gap_range = np.linspace(-10, 10, 50)
inflation_grid, output_gap_grid = np.meshgrid(inflation_range, output_gap_range)

# Prepare grid data for the neural network
grid_points = np.vstack([inflation_grid.ravel(), output_gap_grid.ravel()]).T
grid_points_scaled = scaler_X.transform(grid_points)
grid_tensor = torch.FloatTensor(grid_points_scaled)

# Get predictions from the neural network
ml_model.eval()
with torch.no_grad():
    grid_pred_scaled = ml_model(grid_tensor).numpy().flatten()

grid_pred = scaler_y.inverse_transform(grid_pred_scaled.reshape(-1, 1)).flatten()

# Reshape predictions back to grid
federal_funds_grid = grid_pred.reshape(inflation_grid.shape)

# Create a 3D surface plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(inflation_grid, output_gap_grid, federal_funds_grid, 
                          cmap='viridis', alpha=0.8, linewidth=0)

# Add colorbar
fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)

# Add actual data points for comparison
actual_inflation = data['inflation'].values
actual_output_gap = data['output_gap'].values
actual_ffr = data['actual_rate'].values
ax.scatter(actual_inflation, actual_output_gap, actual_ffr, c='red', s=10, alpha=0.5)

# Add axes labels
ax.set_xlabel('Inflation Rate')
ax.set_ylabel('Output Gap')
ax.set_zlabel('Federal Funds Rate')
ax.set_title('Neural Network Model Surface: Inflation & Output Gap to Federal Funds Rate')

plt.tight_layout()
plt.savefig('neural_network_surface_pytorch.png')
plt.close()

# Create a plot showing how the two different ML models respond to different inputs
plt.figure(figsize=(15, 12))

# 1. Response to different inflation rates (fixing output gap at 0)
plt.subplot(2, 2, 1)
test_inflation = np.linspace(0, 10, 100)
test_output_gap = np.zeros_like(test_inflation)

# Original Taylor Rule
taylor_original_response = original_taylor_rule(test_inflation, test_output_gap)

# OLS Taylor Rule
taylor_ols_response = ols_taylor_rule(test_inflation, test_output_gap)

# ML Taylor Rule
test_data = np.column_stack([test_inflation, test_output_gap])
test_data_scaled = scaler_X.transform(test_data)
test_tensor = torch.FloatTensor(test_data_scaled)

ml_model.eval()
with torch.no_grad():
    ml_response_scaled = ml_model(test_tensor).numpy().flatten()
ml_response = scaler_y.inverse_transform(ml_response_scaled.reshape(-1, 1)).flatten()

plt.plot(test_inflation, taylor_original_response, 'r--', label='Original Taylor Rule')
plt.plot(test_inflation, taylor_ols_response, 'g-.', label='OLS Taylor Rule')
plt.plot(test_inflation, ml_response, 'b-', label='ML Taylor Rule')
plt.xlabel('Inflation Rate')
plt.ylabel('Federal Funds Rate')
plt.title('Response to Inflation (Output Gap = 0)')
plt.legend()

# 2. Response to different output gaps (fixing inflation at 2%)
plt.subplot(2, 2, 2)
test_output_gap = np.linspace(-6, 6, 100)
test_inflation = np.ones_like(test_output_gap) * 2.0

# Original Taylor Rule
taylor_original_response = original_taylor_rule(test_inflation, test_output_gap)

# OLS Taylor Rule
taylor_ols_response = ols_taylor_rule(test_inflation, test_output_gap)

# ML Taylor Rule
test_data = np.column_stack([test_inflation, test_output_gap])
test_data_scaled = scaler_X.transform(test_data)
test_tensor = torch.FloatTensor(test_data_scaled)

with torch.no_grad():
    ml_response_scaled = ml_model(test_tensor).numpy().flatten()
ml_response = scaler_y.inverse_transform(ml_response_scaled.reshape(-1, 1)).flatten()

plt.plot(test_output_gap, taylor_original_response, 'r--', label='Original Taylor Rule')
plt.plot(test_output_gap, taylor_ols_response, 'g-.', label='OLS Taylor Rule')
plt.plot(test_output_gap, ml_response, 'b-', label='ML Taylor Rule')
plt.xlabel('Output Gap')
plt.ylabel('Federal Funds Rate')
plt.title('Response to Output Gap (Inflation = 2%)')
plt.legend()

# 3. Response to different inflation rates during bubble bursts
plt.subplot(2, 2, 3)
test_inflation = np.linspace(0, 10, 100)
test_output_gap = np.zeros_like(test_inflation)
test_bubble = np.ones_like(test_inflation)

# ML Taylor Rule with bubble indicator
test_data_improved = np.column_stack([test_inflation, test_output_gap, test_bubble])
test_data_improved_scaled = StandardScaler().fit_transform(test_data_improved)
test_improved_tensor = torch.FloatTensor(test_data_improved_scaled)

ml_model_improved.eval()
with torch.no_grad():
    ml_improved_response_scaled = ml_model_improved(test_improved_tensor).numpy().flatten()
ml_improved_response = scaler_y.inverse_transform(ml_improved_response_scaled.reshape(-1, 1)).flatten()

# ML Taylor Rule without bubble indicator
test_data = np.column_stack([test_inflation, test_output_gap])
test_data_scaled = scaler_X.transform(test_data)
test_tensor = torch.FloatTensor(test_data_scaled)

with torch.no_grad():
    ml_response_scaled = ml_model(test_tensor).numpy().flatten()
ml_response = scaler_y.inverse_transform(ml_response_scaled.reshape(-1, 1)).flatten()

plt.plot(test_inflation, ml_response, 'b-', label='ML Taylor Rule')
plt.plot(test_inflation, ml_improved_response, 'm-', label='ML with Bubble Indicator')
plt.xlabel('Inflation Rate')
plt.ylabel('Federal Funds Rate')
plt.title('Response to Inflation During Bubble Bursts (Output Gap = 0)')
plt.legend()

# 4. Response to different output gaps during bubble bursts
plt.subplot(2, 2, 4)
test_output_gap = np.linspace(-6, 6, 100)
test_inflation = np.ones_like(test_output_gap) * 2.0
test_bubble = np.ones_like(test_output_gap)

# ML Taylor Rule with bubble indicator
test_data_improved = np.column_stack([test_inflation, test_output_gap, test_bubble])
test_data_improved_scaled = StandardScaler().fit_transform(test_data_improved)
test_improved_tensor = torch.FloatTensor(test_data_improved_scaled)

with torch.no_grad():
    ml_improved_response_scaled = ml_model_improved(test_improved_tensor).numpy().flatten()
ml_improved_response = scaler_y.inverse_transform(ml_improved_response_scaled.reshape(-1, 1)).flatten()

# ML Taylor Rule without bubble indicator
test_data = np.column_stack([test_inflation, test_output_gap])
test_data_scaled = scaler_X.transform(test_data)
test_tensor = torch.FloatTensor(test_data_scaled)

with torch.no_grad():
    ml_response_scaled = ml_model(test_tensor).numpy().flatten()
ml_response = scaler_y.inverse_transform(ml_response_scaled.reshape(-1, 1)).flatten()

plt.plot(test_output_gap, ml_response, 'b-', label='ML Taylor Rule')
plt.plot(test_output_gap, ml_improved_response, 'm-', label='ML with Bubble Indicator')
plt.xlabel('Output Gap')
plt.ylabel('Federal Funds Rate')
plt.title('Response to Output Gap During Bubble Bursts (Inflation = 2%)')
plt.legend()

plt.tight_layout()
plt.savefig('model_responses_pytorch.png')
plt.close()

# Final comparison: all models in one plot for full timeline
plt.figure(figsize=(15, 8))
plt.plot(data['date'], data['actual_rate'], 'k-', linewidth=2, label='Actual Federal Funds Rate')
plt.plot(data['date'], data['taylor_original'], 'r--', label='Original Taylor Rule')
plt.plot(data['date'], data['taylor_ols'], 'g-.', label='OLS Taylor Rule')
plt.plot(data['date'], data['taylor_ml'], 'b:', label='ML Taylor Rule')
plt.plot(data['date'], data['taylor_ml_improved'], 'm-', alpha=0.7, label='ML with Bubble Indicator')

# Highlight bubble burst periods
for start, end in bubble_periods:
    if start < len(data):
        end = min(end, len(data))
        plt.axvspan(data['date'][start], data['date'][end], alpha=0.2, color='gray')

plt.title('Comparison of All Taylor Rule Methods')
plt.xlabel('Date')
plt.ylabel('Federal Funds Rate (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('all_models_comparison_pytorch.png')
plt.close()

print("\nAnalysis complete! All models have been implemented using PyTorch.")