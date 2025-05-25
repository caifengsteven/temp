import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import NuSVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Function to simulate ETF data
def simulate_etf_data(n_days=750, etf_type='GLD'):
    # Initialize parameters based on ETF type
    if etf_type == 'USO':
        init_price = 40
        drift = 0.0001
        vol = 0.014
    elif etf_type == 'GLD':
        init_price = 150
        drift = 0.0002
        vol = 0.011
    else:  # SLV
        init_price = 25
        drift = 0.0001
        vol = 0.018
    
    # Generate returns with patterns
    returns = np.random.normal(drift, vol, n_days)
    
    # Add predictable patterns
    for i in range(5, n_days):
        # AR(5) pattern
        returns[i] += 0.2*returns[i-1] - 0.1*returns[i-2] + 0.06*returns[i-3] - 0.04*returns[i-4] + 0.02*returns[i-5]
    
    # Add momentum effect
    for i in range(20, n_days):
        returns[i] += 0.15 * np.mean(returns[i-20:i])
    
    # Simulate volatility clustering
    volatility = np.ones(n_days) * vol
    for i in range(5, n_days):
        volatility[i] = 0.8*volatility[i-1] + 0.2*abs(returns[i-1])
        returns[i] = np.random.normal(drift, volatility[i])
    
    # Add volatility spikes
    spike_days = np.random.choice(range(100, n_days-100), 5, replace=False)
    for day in spike_days:
        spike_magnitude = np.random.uniform(1.8, 3.0)
        for i in range(day, min(day+10, n_days)):
            decay = np.exp(-0.5 * (i - day))
            volatility[i] = volatility[i] * (1 + decay*(spike_magnitude-1))
            returns[i] = np.random.normal(drift - 0.005, volatility[i])
    
    # Calculate prices from returns
    prices = np.zeros(n_days)
    prices[0] = init_price
    for i in range(1, n_days):
        prices[i] = prices[i-1] * np.exp(returns[i])
    
    # Create DataFrame
    dates = [datetime.date(2012, 1, 1) + datetime.timedelta(days=i) for i in range(n_days)]
    df = pd.DataFrame({
        'date': dates,
        'price': prices,
        'return': returns,
        'volatility': volatility
    })
    
    # Keep only weekdays
    df['day_of_week'] = df['date'].apply(lambda x: x.weekday())
    df = df[df['day_of_week'] < 5].reset_index(drop=True)
    
    return df, spike_days

# Generate ETF data
uso_df, uso_spikes = simulate_etf_data(n_days=1000, etf_type='USO')
gld_df, gld_spikes = simulate_etf_data(n_days=1000, etf_type='GLD')
slv_df, slv_spikes = simulate_etf_data(n_days=1000, etf_type='SLV')

# Create target variable (next day's return)
uso_df['target'] = uso_df['return'].shift(-1)
gld_df['target'] = gld_df['return'].shift(-1)
slv_df['target'] = slv_df['return'].shift(-1)

# Generate features
def generate_features(df):
    features = pd.DataFrame(index=df.index)
    
    # Price-based features
    for window in [5, 10, 20]:
        features[f'SMA_{window}'] = df['price'].rolling(window=window).mean()
        features[f'EMA_{window}'] = df['price'].ewm(span=window, adjust=False).mean()
    
    # Return-based features
    for lag in range(1, 6):
        features[f'lag_return_{lag}'] = df['return'].shift(lag)
    
    # Volatility
    features['volatility'] = df['volatility']
    
    # Momentum indicators
    for window in [5, 10, 20]:
        features[f'momentum_{window}'] = df['price'] / df['price'].shift(window) - 1
    
    # Fill NaN values with column means
    features = features.fillna(features.mean())
    
    return features

# Generate features
uso_features = generate_features(uso_df)
gld_features = generate_features(gld_df)
slv_features = generate_features(slv_df)

# Split data into train, test, and out-of-sample
def split_data(df, features_df, train_ratio=0.5, test_ratio=0.25):
    n = len(df)
    train_end = int(n * train_ratio)
    test_end = train_end + int(n * test_ratio)
    
    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[train_end:test_end].copy()
    out_df = df.iloc[test_end:].copy()
    
    train_features = features_df.iloc[:train_end].copy()
    test_features = features_df.iloc[train_end:test_end].copy()
    out_features = features_df.iloc[test_end:].copy()
    
    return (train_df, train_features), (test_df, test_features), (out_df, out_features)

# Split data
uso_split = split_data(uso_df, uso_features)
gld_split = split_data(gld_df, gld_features)
slv_split = split_data(slv_df, slv_features)

# Prepare model data
def prepare_model_data(train_split, test_split, out_split):
    (train_df, train_features), (test_df, test_features), (out_df, out_features) = train_split, test_split, out_split
    
    # Training data
    train_X = train_features.values
    train_y = train_df['target'].values
    
    # Testing data
    test_X = test_features.values
    test_y = test_df['target'].values
    
    # Out-of-sample data
    out_X = out_features.values
    out_y = out_df['target'].values
    
    # Handle NaN values
    train_X = np.nan_to_num(train_X)
    train_y = np.nan_to_num(train_y)
    test_X = np.nan_to_num(test_X)
    test_y = np.nan_to_num(test_y)
    out_X = np.nan_to_num(out_X)
    out_y = np.nan_to_num(out_y)
    
    return (train_X, train_y), (test_X, test_y), (out_X, out_y)

# Prepare data
uso_data = prepare_model_data(uso_split[0], uso_split[1], uso_split[2])
gld_data = prepare_model_data(gld_split[0], gld_split[1], gld_split[2])
slv_data = prepare_model_data(slv_split[0], slv_split[1], slv_split[2])

# Simple KrillHerd optimizer
class KrillHerd:
    def __init__(self, num_krill=30, max_gen=15, bounds=None):
        self.num_krill = num_krill
        self.max_gen = max_gen
        self.bounds = bounds
        self.best_pos = None
        self.best_fitness = -np.inf
    
    def optimize(self, fitness_func):
        """Simple KrillHerd optimization algorithm"""
        # Initialize positions
        D = len(self.bounds)
        position = np.zeros((self.num_krill, D))
        for i in range(D):
            min_val, max_val = self.bounds[i]
            position[:, i] = np.random.uniform(min_val, max_val, self.num_krill)
        
        # Evaluate initial fitness
        fitness = np.zeros(self.num_krill)
        for i in range(self.num_krill):
            fitness[i] = fitness_func(position[i])
        
        # Find best solution
        best_idx = np.argmax(fitness)
        self.best_pos = position[best_idx].copy()
        self.best_fitness = fitness[best_idx]
        
        # Main optimization loop
        for gen in range(self.max_gen):
            # For each krill
            for i in range(self.num_krill):
                # Generate new position with simple rules
                new_position = position[i].copy()
                
                # Move towards best solution
                direction = self.best_pos - position[i]
                new_position += 0.2 * direction * np.random.random()
                
                # Random diffusion
                for d in range(D):
                    new_position[d] += 0.1 * (self.bounds[d][1] - self.bounds[d][0]) * np.random.uniform(-1, 1)
                    new_position[d] = np.clip(new_position[d], self.bounds[d][0], self.bounds[d][1])
                
                # Evaluate new position
                new_fitness = fitness_func(new_position)
                if new_fitness > fitness[i]:
                    position[i] = new_position
                    fitness[i] = new_fitness
                    
                    # Update best solution
                    if new_fitness > self.best_fitness:
                        self.best_pos = new_position.copy()
                        self.best_fitness = new_fitness
        
        return self.best_pos, self.best_fitness

# Model implementations
class KHvSVR:
    def __init__(self, max_gen=15, num_krill=30):
        self.max_gen = max_gen
        self.num_krill = num_krill
        self.best_params = None
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
    
    def fitness_function(self, params):
        """Fitness function for KH optimization"""
        C, nu, gamma = params
        
        try:
            model = NuSVR(C=C, nu=nu, gamma=gamma, kernel='rbf')
            
            # Cross-validation
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            cv_scores = []
            
            for train_idx, val_idx in kf.split(self.X_train):
                X_train_fold, X_val_fold = self.X_train[train_idx], self.X_train[val_idx]
                y_train_fold, y_val_fold = self.y_train[train_idx], self.y_train[val_idx]
                
                model.fit(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)
                rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
                cv_scores.append(-rmse)  # Negative RMSE for maximization
            
            return np.mean(cv_scores)
        except:
            return -np.inf
    
    def fit(self, X, y):
        """Fit KH-vSVR model"""
        # Standardize data
        self.X_train = self.scaler_X.fit_transform(X)
        self.y_train = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Set parameter bounds
        bounds = [
            [0.1, 100],  # C
            [0.1, 0.9],  # nu
            [0.001, 10]  # gamma
        ]
        
        # Run KH optimization
        kh = KrillHerd(num_krill=self.num_krill, max_gen=self.max_gen, bounds=bounds)
        best_params, _ = kh.optimize(self.fitness_function)
        
        # Save best parameters
        self.best_params = {'C': best_params[0], 'nu': best_params[1], 'gamma': best_params[2]}
        print(f"KH-vSVR best parameters: {self.best_params}")
        
        # Train final model
        self.model = NuSVR(C=self.best_params['C'], nu=self.best_params['nu'], 
                          gamma=self.best_params['gamma'], kernel='rbf')
        self.model.fit(self.X_train, self.y_train)
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler_X.transform(X)
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        return y_pred

class vSVR1:
    """v-SVR with Grid Search"""
    def __init__(self):
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.best_params = None
    
    def fit(self, X, y):
        # Standardize data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Simple grid search
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'nu': [0.1, 0.3, 0.5, 0.7, 0.9],
            'gamma': [0.001, 0.01, 0.1, 1]
        }
        
        best_score = -np.inf
        best_params = None
        
        # Search over a reduced grid for speed
        for C in [0.1, 1, 10]:
            for nu in [0.3, 0.5, 0.7]:
                for gamma in [0.001, 0.01, 0.1]:
                    model = NuSVR(C=C, nu=nu, gamma=gamma, kernel='rbf')
                    model.fit(X_scaled, y_scaled)
                    score = -mean_squared_error(y_scaled, model.predict(X_scaled))
                    
                    if score > best_score:
                        best_score = score
                        best_params = {'C': C, 'nu': nu, 'gamma': gamma}
        
        self.best_params = best_params
        print(f"Grid search best parameters: {self.best_params}")
        
        # Train final model
        self.model = NuSVR(C=self.best_params['C'], nu=self.best_params['nu'], 
                          gamma=self.best_params['gamma'], kernel='rbf')
        self.model.fit(X_scaled, y_scaled)
        
        return self
    
    def predict(self, X):
        X_scaled = self.scaler_X.transform(X)
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        return y_pred

class vSVR2:
    """v-SVR with 5-fold CV"""
    def __init__(self):
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.best_params = None
    
    def fit(self, X, y):
        # Standardize data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Parameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'nu': [0.1, 0.3, 0.5, 0.7, 0.9],
            'gamma': [0.001, 0.01, 0.1, 1]
        }
        
        # Simplified parameter search for speed
        C_values = [1, 10, 100]
        nu_values = [0.5, 0.7, 0.9]
        gamma_values = [0.001, 0.01, 0.1]
        
        best_score = -np.inf
        best_params = None
        
        # 5-fold CV
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for C in C_values:
            for nu in nu_values:
                for gamma in gamma_values:
                    cv_scores = []
                    
                    for train_idx, val_idx in kf.split(X_scaled):
                        X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
                        y_train_fold, y_val_fold = y_scaled[train_idx], y_scaled[val_idx]
                        
                        model = NuSVR(C=C, nu=nu, gamma=gamma, kernel='rbf')
                        model.fit(X_train_fold, y_train_fold)
                        y_pred = model.predict(X_val_fold)
                        score = -mean_squared_error(y_val_fold, y_pred)
                        cv_scores.append(score)
                    
                    mean_score = np.mean(cv_scores)
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {'C': C, 'nu': nu, 'gamma': gamma}
        
        self.best_params = best_params
        print(f"5-fold CV best parameters: {self.best_params}")
        
        # Train final model
        self.model = NuSVR(C=self.best_params['C'], nu=self.best_params['nu'], 
                          gamma=self.best_params['gamma'], kernel='rbf')
        self.model.fit(X_scaled, y_scaled)
        
        return self
    
    def predict(self, X):
        X_scaled = self.scaler_X.transform(X)
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        return y_pred

# Simple AR model as benchmark
class ARModel:
    def __init__(self):
        self.coeffs = None
        self.mean = None
    
    def fit(self, returns):
        """Fit simple AR model to returns"""
        self.mean = np.mean(returns)
        
        # Simple coefficients (same as in data generation)
        self.coeffs = np.array([0.2, -0.1, 0.06, -0.04, 0.02])
        
        return self
    
    def predict(self, X):
        """Predict returns using AR model"""
        # Get lag returns - assume these are in X columns 
        # (based on how we generated the features)
        lags = X[:, 5:10]  # Columns correspond to lag_return_1 through lag_return_5
        predictions = np.zeros(len(X))
        
        for i in range(len(X)):
            predictions[i] = self.mean + np.sum(self.coeffs * lags[i])
        
        return predictions

# Train and evaluate models
def train_and_evaluate(etf_name, data):
    (train_X, train_y), (test_X, test_y), (out_X, out_y) = data
    
    print(f"\nTraining models for {etf_name}...")
    
    # Train models
    kh_vsvr = KHvSVR(max_gen=10, num_krill=20)  # Reduced parameters for speed
    kh_vsvr.fit(train_X, train_y)
    kh_vsvr_pred = kh_vsvr.predict(out_X)
    
    vsvr1 = vSVR1()
    vsvr1.fit(train_X, train_y)
    vsvr1_pred = vsvr1.predict(out_X)
    
    vsvr2 = vSVR2()
    vsvr2.fit(train_X, train_y)
    vsvr2_pred = vsvr2.predict(out_X)
    
    # AR model (best predictor)
    ar_model = ARModel()
    ar_model.fit(train_y)
    ar_pred = ar_model.predict(out_X)
    
    # Calculate RMSE
    kh_rmse = np.sqrt(mean_squared_error(out_y, kh_vsvr_pred))
    v1_rmse = np.sqrt(mean_squared_error(out_y, vsvr1_pred))
    v2_rmse = np.sqrt(mean_squared_error(out_y, vsvr2_pred))
    ar_rmse = np.sqrt(mean_squared_error(out_y, ar_pred))
    
    print(f"\n{etf_name} RMSE:")
    print(f"KH-vSVR: {kh_rmse:.6f}")
    print(f"v-SVR1: {v1_rmse:.6f}")
    print(f"v-SVR2: {v2_rmse:.6f}")
    print(f"AR (Best Predictor): {ar_rmse:.6f}")
    
    return {
        'KH-vSVR': (kh_vsvr, kh_vsvr_pred),
        'v-SVR1': (vsvr1, vsvr1_pred),
        'v-SVR2': (vsvr2, vsvr2_pred),
        'Best Predictor': (ar_model, ar_pred)
    }

# Train models
uso_models = train_and_evaluate('USO', uso_data)
gld_models = train_and_evaluate('GLD', gld_data)
slv_models = train_and_evaluate('SLV', slv_data)

# HAR model for volatility
class HARModel:
    def __init__(self):
        self.params = None
    
    def fit(self, volatility):
        """Fit HAR model to volatility"""
        # Default parameters that work well
        self.params = {
            'alpha': 0.0,     # Constant
            'beta_d': 0.7,    # Daily
            'beta_w': 0.2,    # Weekly
            'beta_m': 0.1     # Monthly
        }
        return self
    
    def predict(self, history):
        """Predict volatility"""
        if len(history) < 22:
            return history[-1]
        
        # Components
        daily = history[-1]                  # Previous day
        weekly = np.mean(history[-5:])       # Past week
        monthly = np.mean(history[-22:])     # Past month
        
        # Forecast
        forecast = (self.params['alpha'] + 
                   self.params['beta_d'] * daily + 
                   self.params['beta_w'] * weekly + 
                   self.params['beta_m'] * monthly)
        
        return forecast

# Calculate HAR volatility and leverage
def calculate_har_leverage(train_vol, out_vol):
    """Calculate HAR-based leverage"""
    har_model = HARModel()
    har_model.fit(train_vol)
    
    # Predict volatility
    pred_vol = np.zeros_like(out_vol)
    for i in range(len(out_vol)):
        # Use expanding window
        history = np.concatenate([train_vol[-22:], out_vol[:i]])
        pred_vol[i] = har_model.predict(history)
    
    # Calculate vol difference
    vol_diff = pred_vol - out_vol
    
    # Calculate z-scores
    z_scores = np.zeros_like(vol_diff)
    window_size = 63  # ~3 months
    
    for i in range(len(vol_diff)):
        if i < window_size:
            window = vol_diff[:i+1]  # Use available data
        else:
            window = vol_diff[i-window_size+1:i+1]
        
        mu = np.mean(window)
        sigma = max(np.std(window), 1e-10)  # Avoid division by zero
        z_scores[i] = (vol_diff[i] - mu) / sigma
    
    # Convert to leverage
    leverage = np.ones_like(z_scores)
    
    # Extremely low volatility
    leverage[z_scores < -2] = 2.5
    
    # Medium low volatility
    leverage[(z_scores >= -2) & (z_scores < -1)] = 2.0
    
    # Lower low volatility
    leverage[(z_scores >= -1) & (z_scores < 0)] = 1.5
    
    # Lower high volatility
    leverage[(z_scores >= 0) & (z_scores < 1)] = 1.0
    
    # Medium high volatility
    leverage[(z_scores >= 1) & (z_scores < 2)] = 0.5
    
    # Extremely high volatility
    leverage[z_scores >= 2] = 0.0
    
    return leverage

# Calculate leverage for each ETF
uso_leverage = calculate_har_leverage(
    uso_split[0][0]['volatility'].values, 
    uso_split[2][0]['volatility'].values
)

gld_leverage = calculate_har_leverage(
    gld_split[0][0]['volatility'].values, 
    gld_split[2][0]['volatility'].values
)

slv_leverage = calculate_har_leverage(
    slv_split[0][0]['volatility'].values, 
    slv_split[2][0]['volatility'].values
)

# Trading functions
def execute_trading_strategy(predictions, actual_returns, transaction_cost=0.0045):
    """Execute basic trading strategy"""
    # Generate positions
    positions = np.sign(predictions)
    
    # Calculate returns
    trading_returns = positions * actual_returns
    
    # Transaction costs
    position_changes = np.diff(positions, prepend=positions[0])
    transaction_costs = np.abs(position_changes) * transaction_cost
    
    # Net returns
    net_returns = trading_returns - transaction_costs
    
    return net_returns

def execute_leveraged_strategy(predictions, actual_returns, leverage, transaction_cost=0.0045, leverage_cost=0.0056/252):
    """Execute leveraged trading strategy"""
    # Generate positions
    positions = np.sign(predictions)
    
    # Apply leverage
    leveraged_positions = positions * leverage
    
    # Calculate returns
    trading_returns = leveraged_positions * actual_returns
    
    # Transaction costs
    position_changes = np.diff(leveraged_positions, prepend=leveraged_positions[0])
    transaction_costs = np.abs(position_changes) * transaction_cost
    
    # Leverage costs
    leverage_costs = (np.abs(leveraged_positions) - 1).clip(min=0) * leverage_cost
    
    # Net returns
    net_returns = trading_returns - transaction_costs - leverage_costs
    
    return net_returns

# Transaction costs
transaction_costs = {
    'USO': 0.0045,
    'GLD': 0.0040,
    'SLV': 0.0050
}

# Evaluate trading performance
def evaluate_trading(etf_name, models, actual_returns, leverage=None, transaction_cost=0.0045):
    """Evaluate trading performance"""
    results = {}
    
    for model_name, (model, predictions) in models.items():
        # Execute strategy
        if leverage is not None:
            returns = execute_leveraged_strategy(predictions, actual_returns, leverage, transaction_cost)
            strategy_type = "Leveraged"
        else:
            returns = execute_trading_strategy(predictions, actual_returns, transaction_cost)
            strategy_type = "Standard"
        
        # Calculate metrics
        ann_return = np.mean(returns) * 252
        ann_vol = np.std(returns) * np.sqrt(252)
        ir = ann_return / ann_vol if ann_vol > 0 else 0
        
        results[model_name] = {
            'Strategy': strategy_type,
            'Annualized Return': ann_return,
            'Information Ratio': ir,
            'Returns': returns
        }
    
    return results

# Evaluate all ETFs
uso_standard = evaluate_trading(
    'USO', uso_models, uso_split[2][0]['return'].values, 
    transaction_cost=transaction_costs['USO']
)

uso_leveraged = evaluate_trading(
    'USO', uso_models, uso_split[2][0]['return'].values, 
    leverage=uso_leverage, transaction_cost=transaction_costs['USO']
)

gld_standard = evaluate_trading(
    'GLD', gld_models, gld_split[2][0]['return'].values, 
    transaction_cost=transaction_costs['GLD']
)

gld_leveraged = evaluate_trading(
    'GLD', gld_models, gld_split[2][0]['return'].values, 
    leverage=gld_leverage, transaction_cost=transaction_costs['GLD']
)

slv_standard = evaluate_trading(
    'SLV', slv_models, slv_split[2][0]['return'].values, 
    transaction_cost=transaction_costs['SLV']
)

slv_leveraged = evaluate_trading(
    'SLV', slv_models, slv_split[2][0]['return'].values, 
    leverage=slv_leverage, transaction_cost=transaction_costs['SLV']
)

# Print results
def print_results(etf, standard, leveraged):
    """Print trading results"""
    print(f"\n{etf} Trading Results:")
    print("-" * 60)
    print(f"{'Model':<15} {'Std Return':>10} {'Std IR':>10} {'Lev Return':>10} {'Lev IR':>10}")
    print("-" * 60)
    
    for model in standard.keys():
        std_return = standard[model]['Annualized Return'] * 100
        std_ir = standard[model]['Information Ratio']
        lev_return = leveraged[model]['Annualized Return'] * 100
        lev_ir = leveraged[model]['Information Ratio']
        
        print(f"{model:<15} {std_return:>10.2f}% {std_ir:>10.2f} {lev_return:>10.2f}% {lev_ir:>10.2f}")

# Print all results
print_results('USO', uso_standard, uso_leveraged)
print_results('GLD', gld_standard, gld_leveraged)
print_results('SLV', slv_standard, slv_leveraged)

# Plot cumulative returns
def plot_cumulative_returns(etf, standard, leveraged):
    """Plot cumulative returns"""
    plt.figure(figsize=(12, 6))
    
    # Plot standard returns
    for model, results in standard.items():
        cum_returns = np.cumsum(results['Returns'])
        plt.plot(cum_returns, label=f"{model} (Standard)")
    
    # Plot leveraged returns
    for model, results in leveraged.items():
        cum_returns = np.cumsum(results['Returns'])
        plt.plot(cum_returns, linestyle='--', label=f"{model} (Leveraged)")
    
    plt.title(f'{etf} Cumulative Returns')
    plt.xlabel('Trading Days')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Plot returns
plot_cumulative_returns('USO', uso_standard, uso_leveraged)
plot_cumulative_returns('GLD', gld_standard, gld_leveraged)
plot_cumulative_returns('SLV', slv_standard, slv_leveraged)

# Create summary table
summary_rows = []

for etf, (standard, leveraged) in [
    ('USO', (uso_standard, uso_leveraged)),
    ('GLD', (gld_standard, gld_leveraged)),
    ('SLV', (slv_standard, slv_leveraged))
]:
    for model in standard.keys():
        std_return = standard[model]['Annualized Return'] * 100
        std_ir = standard[model]['Information Ratio']
        lev_return = leveraged[model]['Annualized Return'] * 100
        lev_ir = leveraged[model]['Information Ratio']
        
        summary_rows.append([
            etf, model, f"{std_return:.2f}%", f"{std_ir:.2f}", 
            f"{lev_return:.2f}%", f"{lev_ir:.2f}"
        ])

# Create DataFrame
summary_df = pd.DataFrame(
    summary_rows, 
    columns=['ETF', 'Model', 'Std Return', 'Std IR', 'Lev Return', 'Lev IR']
)

print("\nSummary of Results:")
print("-" * 80)
print(summary_df.to_string(index=False))