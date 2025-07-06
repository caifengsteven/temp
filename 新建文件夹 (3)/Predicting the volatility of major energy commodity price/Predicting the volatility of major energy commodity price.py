import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)

class TVPARModel:
    """
    Time-Varying Parameter Autoregressive Model
    """
    def __init__(self, p=2, kernel_width=0.3):
        self.p = p
        self.kernel_width = kernel_width
        self.coef_history = []
        
    def _local_kernel_weights(self, t, T):
        """Calculate local kernel weights for time point t/T"""
        u = np.linspace(0, 1, T)
        weights = np.exp(-0.5 * ((u - t/T) / self.kernel_width)**2)
        weights = weights / np.sum(weights)
        return weights
    
    def fit(self, y):
        """Fit TV-AR model using local linear regression"""
        T = len(y)
        self.coef = np.zeros((T, self.p + 1))  # +1 for intercept
        
        for t in range(self.p, T):
            # Get kernel weights centered at t/T
            weights = self._local_kernel_weights(t, T)
            
            # Prepare lagged data
            X = np.zeros((T - self.p, self.p + 1))
            X[:, 0] = 1  # Intercept
            for i in range(self.p):
                X[:, i + 1] = y[self.p - i - 1:T - i - 1]
            
            # Target variable
            y_target = y[self.p:T]
            
            # Weighted least squares
            W = np.diag(weights[self.p:T])
            XtW = X.T @ W
            beta = np.linalg.inv(XtW @ X) @ XtW @ y_target
            
            self.coef[t, :] = beta
            
        return self
    
    def predict(self, y, h=1):
        """Make h-step ahead predictions"""
        T = len(y)
        pred = np.zeros(T - self.p)
        
        for t in range(self.p, T):
            X = np.concatenate([[1], y[t-self.p:t][::-1]])
            pred[t-self.p] = X @ self.coef[t, :]
            
        return pred
    
    def compute_residuals(self, y):
        """Compute residuals"""
        T = len(y)
        residuals = np.zeros(T - self.p)
        
        for t in range(self.p, T):
            X = np.concatenate([[1], y[t-self.p:t][::-1]])
            residuals[t-self.p] = y[t] - X @ self.coef[t, :]
            
        return residuals
    
    def compute_alpha(self, max_lag=100):
        """Compute time-varying MA(∞) coefficients"""
        T = len(self.coef)
        alpha = np.zeros((T, max_lag + 1))
        
        for t in range(self.p, T):
            ar_coef = self.coef[t, 1:]
            # Convert AR coefficients to MA coefficients
            alpha[t, 0] = 1
            for i in range(1, max_lag + 1):
                if i <= self.p:
                    alpha[t, i] = ar_coef[i-1]
                else:
                    alpha[t, i] = np.sum([ar_coef[j] * alpha[t, i-j-1] for j in range(min(self.p, i))])
                    
        return alpha


class TVEWDModel:
    """
    Time-Varying Extended Wold Decomposition Model
    """
    def __init__(self, p=2, J=7, kernel_width=0.3):
        self.p = p
        self.J = J  # Number of scales
        self.kernel_width = kernel_width
        self.tvpar = TVPARModel(p=p, kernel_width=kernel_width)
        
    def fit(self, y):
        """Fit the TV-EWD model"""
        self.tvpar.fit(y)
        self.y = y
        self.T = len(y)
        
        # Compute residuals (innovations)
        self.epsilon = self.tvpar.compute_residuals(y)
        
        # Compute time-varying MA(∞) coefficients
        self.alpha = self.tvpar.compute_alpha(max_lag=2**self.J)
        
        # Compute scale-specific coefficients beta and innovations
        self.beta = {}
        self.scale_epsilon = {}
        
        for j in range(1, self.J + 1):
            # Compute scale-specific coefficients
            self.beta[j] = np.zeros((self.T, 2**self.J))
            
            for t in range(self.p, self.T):
                for k in range(2**self.J // 2**j):
                    # Compute beta coefficients as per Eq. (2) in the paper
                    self.beta[j][t, k] = (1 / np.sqrt(2**j)) * (
                        np.sum(self.alpha[t, k * 2**j + np.arange(2**(j-1))]) - 
                        np.sum(self.alpha[t, k * 2**j + 2**(j-1) + np.arange(2**(j-1))])
                    )
            
            # Compute scale-specific innovations
            self.scale_epsilon[j] = np.zeros(self.T - self.p)
            epsilon_padded = np.pad(self.epsilon, (0, 2**j), 'constant')
            
            for t in range(self.p, self.T):
                idx = t - self.p
                # Compute epsilon_j as per the paper
                self.scale_epsilon[j][idx] = (1 / np.sqrt(2**j)) * (
                    np.sum(epsilon_padded[idx:idx + 2**(j-1)]) - 
                    np.sum(epsilon_padded[idx + 2**(j-1):idx + 2**j])
                )
        
        # Compute scale components
        self.scale_components = {}
        for j in range(1, self.J + 1):
            self.scale_components[j] = np.zeros(self.T - self.p)
            
            for t in range(self.p, self.T):
                idx = t - self.p
                # Compute v_j as per Eq. (3) in the paper
                for k in range(min(idx + 1, 2**self.J // 2**j)):
                    if idx - k * 2**j >= 0:
                        self.scale_components[j][idx] += (
                            self.beta[j][t, k] * self.scale_epsilon[j][idx - k * 2**j]
                        )
        
        # Compute weights for each scale component
        self.weights = self._compute_weights()
        
        return self
    
    def _compute_weights(self):
        """Compute weights for each scale component based on variance contribution"""
        weights = {}
        
        # Compute variance of each scale component
        var_components = {}
        for j in range(1, self.J + 1):
            var_components[j] = np.var(self.scale_components[j])
        
        # Normalize to get weights
        total_var = sum(var_components.values())
        for j in range(1, self.J + 1):
            weights[j] = var_components[j] / total_var
            
        return weights
    
    def predict(self, y, h=1):
        """Make h-step ahead predictions"""
        # For one-step ahead forecasts
        if h == 1:
            # Get the last p values
            last_values = y[-self.p:]
            
            # Predict using TV-AR coefficients
            ar_pred = np.concatenate([[1], last_values[::-1]]) @ self.tvpar.coef[-1, :]
            
            # Weight scale components' contributions
            scale_pred = 0
            for j in range(1, self.J + 1):
                if len(self.scale_components[j]) > 0:
                    scale_pred += self.weights[j] * self.scale_components[j][-1]
            
            return ar_pred + scale_pred
        else:
            # For multi-step ahead forecasts
            # This is a simplified implementation - the paper uses a more complex approach
            forecasts = np.zeros(h)
            temp_y = y.copy()
            
            for i in range(h):
                pred = self.predict(temp_y, h=1)
                forecasts[i] = pred
                temp_y = np.append(temp_y, pred)
                
            return forecasts
    
    def forecast_multiple_steps(self, y, h=1, n_ahead=1):
        """Generate multiple forecasts h steps ahead"""
        forecasts = np.zeros(n_ahead)
        
        for i in range(n_ahead):
            y_temp = y[i:i+len(y)]
            pred = self.predict(y_temp, h=h)
            forecasts[i] = pred[h-1] if h > 1 else pred
            
        return forecasts
    
    def get_persistence_structure(self):
        """Get the time-varying persistence structure"""
        persistence = np.zeros((self.T - self.p, self.J))
        
        for t in range(self.p, self.T):
            idx = t - self.p
            for j in range(1, self.J + 1):
                # Use the first coefficient of the multiscale impulse response as a measure of persistence
                persistence[idx, j-1] = self.beta[j][t, 0]
                
        # Normalize to get relative importance
        persistence_sum = np.sum(np.abs(persistence), axis=1, keepdims=True)
        persistence_normalized = persistence / persistence_sum
        
        return persistence_normalized


class HARModel:
    """
    Heterogeneous Autoregressive (HAR) Model
    """
    def __init__(self):
        pass
    
    def fit(self, y):
        """Fit HAR model"""
        self.y = y
        self.T = len(y)
        
        # Create HAR regressors (daily, weekly, monthly)
        X = np.zeros((self.T - 22, 3))
        for t in range(22, self.T):
            X[t - 22, 0] = y[t - 1]  # Daily
            X[t - 22, 1] = np.mean(y[t - 5:t])  # Weekly
            X[t - 22, 2] = np.mean(y[t - 22:t])  # Monthly
        
        # Target variable
        y_target = y[22:]
        
        # OLS estimation
        self.coef = np.linalg.inv(X.T @ X) @ X.T @ y_target
        
        return self
    
    def predict(self, y, h=1):
        """Make h-step ahead predictions"""
        # For simplicity, we'll use a direct forecasting approach
        if h == 1:
            # Create HAR regressors
            X = np.zeros(3)
            X[0] = y[-1]  # Daily
            X[1] = np.mean(y[-5:])  # Weekly
            X[2] = np.mean(y[-22:])  # Monthly
            
            return X @ self.coef
        else:
            # For multi-step ahead forecasts
            forecasts = np.zeros(h)
            temp_y = y.copy()
            
            for i in range(h):
                pred = self.predict(temp_y, h=1)
                forecasts[i] = pred
                temp_y = np.append(temp_y, pred)
                
            return forecasts
    
    def forecast_multiple_steps(self, y, h=1, n_ahead=1):
        """Generate multiple forecasts h steps ahead"""
        forecasts = np.zeros(n_ahead)
        
        for i in range(n_ahead):
            y_temp = y[i:i+len(y)]
            pred = self.predict(y_temp, h=h)
            forecasts[i] = pred[h-1] if h > 1 else pred
            
        return forecasts


class TVHARModel:
    """
    Time-Varying Parameter Heterogeneous Autoregressive (TV-HAR) Model
    """
    def __init__(self, kernel_width=0.3):
        self.kernel_width = kernel_width
        
    def _local_kernel_weights(self, t, T):
        """Calculate local kernel weights for time point t/T"""
        u = np.linspace(0, 1, T)
        weights = np.exp(-0.5 * ((u - t/T) / self.kernel_width)**2)
        weights = weights / np.sum(weights)
        return weights
    
    def fit(self, y):
        """Fit TV-HAR model using local linear regression"""
        self.y = y
        self.T = len(y)
        
        # Initialize coefficients
        self.coef = np.zeros((self.T - 22, 4))  # +1 for intercept
        
        for t in range(22, self.T):
            # Get kernel weights centered at t/T
            weights = self._local_kernel_weights(t, self.T)
            
            # Create HAR regressors (intercept, daily, weekly, monthly)
            X = np.zeros((self.T - 22, 4))
            X[:, 0] = 1  # Intercept
            
            for i in range(22, self.T):
                X[i - 22, 1] = y[i - 1]  # Daily
                X[i - 22, 2] = np.mean(y[i - 5:i])  # Weekly
                X[i - 22, 3] = np.mean(y[i - 22:i])  # Monthly
            
            # Target variable
            y_target = y[22:]
            
            # Weighted least squares
            W = np.diag(weights[22:self.T])
            XtW = X.T @ W
            beta = np.linalg.inv(XtW @ X) @ XtW @ y_target
            
            self.coef[t - 22, :] = beta
            
        return self
    
    def predict(self, y, h=1):
        """Make h-step ahead predictions"""
        # For simplicity, we'll use a direct forecasting approach
        if h == 1:
            # Create HAR regressors
            X = np.zeros(4)
            X[0] = 1  # Intercept
            X[1] = y[-1]  # Daily
            X[2] = np.mean(y[-5:])  # Weekly
            X[3] = np.mean(y[-22:])  # Monthly
            
            return X @ self.coef[-1, :]
        else:
            # For multi-step ahead forecasts
            forecasts = np.zeros(h)
            temp_y = y.copy()
            
            for i in range(h):
                pred = self.predict(temp_y, h=1)
                forecasts[i] = pred
                temp_y = np.append(temp_y, pred)
                
            return forecasts
    
    def forecast_multiple_steps(self, y, h=1, n_ahead=1):
        """Generate multiple forecasts h steps ahead"""
        forecasts = np.zeros(n_ahead)
        
        for i in range(n_ahead):
            y_temp = y[i:i+len(y)]
            pred = self.predict(y_temp, h=h)
            forecasts[i] = pred[h-1] if h > 1 else pred
            
        return forecasts


class EWDModel:
    """
    Extended Wold Decomposition (EWD) Model
    """
    def __init__(self, p=2, J=7):
        self.p = p
        self.J = J  # Number of scales
        
    def fit(self, y):
        """Fit the EWD model"""
        self.T = len(y)
        
        # Fit AR model
        model = AutoReg(y, lags=self.p, trend='c')
        self.ar_model = model.fit()
        
        # Get residuals
        self.epsilon = self.ar_model.resid[self.p:]
        
        # Get AR coefficients
        ar_coef = self.ar_model.params[1:self.p+1]
        
        # Compute MA(∞) coefficients
        max_lag = 2**self.J
        self.alpha = np.zeros(max_lag + 1)
        self.alpha[0] = 1
        
        for i in range(1, max_lag + 1):
            if i <= self.p:
                self.alpha[i] = ar_coef[i-1]
            else:
                self.alpha[i] = np.sum([ar_coef[j] * self.alpha[i-j-1] for j in range(min(self.p, i))])
        
        # Compute scale-specific coefficients beta and innovations
        self.beta = {}
        self.scale_epsilon = {}
        
        for j in range(1, self.J + 1):
            # Compute scale-specific coefficients
            self.beta[j] = np.zeros(2**self.J)
            
            for k in range(2**self.J // 2**j):
                # Compute beta coefficients as per Eq. (2) in the paper
                self.beta[j][k] = (1 / np.sqrt(2**j)) * (
                    np.sum(self.alpha[k * 2**j + np.arange(2**(j-1))]) - 
                    np.sum(self.alpha[k * 2**j + 2**(j-1) + np.arange(2**(j-1))])
                )
            
            # Compute scale-specific innovations
            self.scale_epsilon[j] = np.zeros(len(self.epsilon))
            epsilon_padded = np.pad(self.epsilon, (0, 2**j), 'constant')
            
            for t in range(len(self.epsilon)):
                # Compute epsilon_j as per the paper
                self.scale_epsilon[j][t] = (1 / np.sqrt(2**j)) * (
                    np.sum(epsilon_padded[t:t + 2**(j-1)]) - 
                    np.sum(epsilon_padded[t + 2**(j-1):t + 2**j])
                )
        
        # Compute scale components
        self.scale_components = {}
        for j in range(1, self.J + 1):
            self.scale_components[j] = np.zeros(len(self.epsilon))
            
            for t in range(len(self.epsilon)):
                # Compute v_j as per Eq. (3) in the paper
                for k in range(min(t + 1, 2**self.J // 2**j)):
                    if t - k * 2**j >= 0:
                        self.scale_components[j][t] += (
                            self.beta[j][k] * self.scale_epsilon[j][t - k * 2**j]
                        )
        
        # Compute weights for each scale component
        self.weights = self._compute_weights()
        
        return self
    
    def _compute_weights(self):
        """Compute weights for each scale component based on variance contribution"""
        weights = {}
        
        # Compute variance of each scale component
        var_components = {}
        for j in range(1, self.J + 1):
            var_components[j] = np.var(self.scale_components[j])
        
        # Normalize to get weights
        total_var = sum(var_components.values())
        for j in range(1, self.J + 1):
            weights[j] = var_components[j] / total_var
            
        return weights
    
    def predict(self, y, h=1):
        """Make h-step ahead predictions"""
        # For one-step ahead forecasts
        if h == 1:
            # Get AR prediction
            ar_pred = self.ar_model.predict(start=len(y), end=len(y))
            
            # Weight scale components' contributions
            scale_pred = 0
            for j in range(1, self.J + 1):
                if len(self.scale_components[j]) > 0:
                    scale_pred += self.weights[j] * self.scale_components[j][-1]
            
            return ar_pred + scale_pred
        else:
            # For multi-step ahead forecasts
            forecasts = np.zeros(h)
            temp_y = y.copy()
            
            for i in range(h):
                pred = self.predict(temp_y, h=1)
                forecasts[i] = pred
                temp_y = np.append(temp_y, pred)
                
            return forecasts
    
    def forecast_multiple_steps(self, y, h=1, n_ahead=1):
        """Generate multiple forecasts h steps ahead"""
        forecasts = np.zeros(n_ahead)
        
        for i in range(n_ahead):
            y_temp = y[i:i+len(y)]
            self.fit(y_temp)  # Refit model for each window
            pred = self.predict(y_temp, h=h)
            forecasts[i] = pred[h-1] if h > 1 else pred
            
        return forecasts


def simulate_energy_volatility(n=2000, show_plot=True):
    """
    Simulate energy volatility with time-varying persistence
    """
    t = np.arange(n)
    
    # Create base volatility with long memory (persistent component)
    d = 0.4  # Long memory parameter
    ar_coef = np.array([0.7, -0.2, 0.1])  # AR coefficients
    
    # Generate fractionally integrated noise
    noise = np.random.normal(0, 1, n+100)
    frac_noise = np.zeros(n+100)
    
    for i in range(1, n+100):
        frac_noise[i] = noise[i]
        for j in range(1, i+1):
            gamma = (-1)**(j+1) * np.prod([(d-k+1)/k for k in range(1, j+1)])
            frac_noise[i] += gamma * noise[i-j]
    
    frac_noise = frac_noise[100:]
    
    # Apply AR structure
    y = np.zeros(n)
    y[:3] = frac_noise[:3]
    
    for i in range(3, n):
        y[i] = np.sum(ar_coef * y[i-3:i][::-1]) + frac_noise[i]
    
    # Add time-varying component (change in persistence)
    # Simulate shifts in persistence at specific time points
    shifts = [int(n*0.25), int(n*0.5), int(n*0.75)]
    
    for shift in shifts:
        if shift == shifts[0]:  # First shift - increase persistence
            ar_coef = np.array([0.85, -0.1, 0.05])
        elif shift == shifts[1]:  # Second shift - decrease persistence
            ar_coef = np.array([0.5, -0.1, 0.1])
        else:  # Third shift - mixed persistence
            ar_coef = np.array([0.7, -0.3, 0.2])
        
        for i in range(shift, min(shift+500, n)):
            # Gradually transition to new persistence
            weight = min((i - shift) / 50, 1.0)
            old_contrib = np.sum(ar_coef * y[i-3:i][::-1]) * (1 - weight)
            new_contrib = np.sum(ar_coef * y[i-3:i][::-1]) * weight
            y[i] = old_contrib + new_contrib + frac_noise[i]
    
    # Scale to typical volatility values
    y = np.abs(y)  # Ensure positivity
    y = 10 * y / np.std(y)  # Scale
    y = y + 20  # Add baseline
    
    # Add jumps and regime-specific volatility
    jumps = np.zeros(n)
    jumps[shifts[0]-50:shifts[0]+50] = 30 * np.random.exponential(0.1, 100)  # Covid-like shock
    jumps[shifts[2]-25:shifts[2]+25] = 20 * np.random.exponential(0.1, 50)   # Geopolitical shock
    
    # Final simulated volatility
    volatility = y + jumps
    
    if show_plot:
        plt.figure(figsize=(12, 6))
        plt.plot(volatility)
        plt.title('Simulated Energy Commodity Volatility')
        plt.xlabel('Time')
        plt.ylabel('Volatility')
        plt.axvline(x=shifts[0], color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=shifts[1], color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=shifts[2], color='r', linestyle='--', alpha=0.5)
        plt.show()
    
    return volatility


def evaluate_models_simulated_data(h_values=[1, 5, 22], window_size=700, n_test=600):
    """
    Evaluate models on simulated data
    """
    # Simulate volatility data
    data = simulate_energy_volatility(n=window_size + n_test)
    
    # Initialize results dictionary
    results = {h: {} for h in h_values}
    
    # Rolling window forecasting
    for h in h_values:
        print(f"\nGenerating {h}-step ahead forecasts...")
        
        # Initialize forecasts
        tvpar_forecasts = np.zeros(n_test)
        tvewd_forecasts = np.zeros(n_test)
        har_forecasts = np.zeros(n_test)
        tvhar_forecasts = np.zeros(n_test)
        ewd_forecasts = np.zeros(n_test)
        
        # True values to compare with
        true_values = data[window_size:window_size + n_test]
        
        # Loop through test period
        for i in tqdm(range(n_test)):
            # Get training window
            train_data = data[i:i + window_size]
            
            # Fit models and make forecasts
            # TV-AR
            tvpar = TVPARModel(p=3, kernel_width=0.3)
            tvpar.fit(train_data)
            tvpar_forecasts[i] = tvpar.predict(train_data, h=h)[-1]
            
            # TV-EWD
            tvewd = TVEWDModel(p=3, J=7, kernel_width=0.3)
            tvewd.fit(train_data)
            tvewd_pred = tvewd.predict(train_data, h=h)
            tvewd_forecasts[i] = tvewd_pred[-1] if h > 1 else tvewd_pred
            
            # HAR
            har = HARModel()
            har.fit(train_data)
            har_forecasts[i] = har.predict(train_data, h=h)[-1] if h > 1 else har.predict(train_data, h=h)
            
            # TV-HAR
            tvhar = TVHARModel(kernel_width=0.3)
            tvhar.fit(train_data)
            tvhar_forecasts[i] = tvhar.predict(train_data, h=h)[-1] if h > 1 else tvhar.predict(train_data, h=h)
            
            # EWD
            ewd = EWDModel(p=3, J=7)
            ewd.fit(train_data)
            ewd_pred = ewd.predict(train_data, h=h)
            ewd_forecasts[i] = ewd_pred[-1] if h > 1 else ewd_pred
        
        # Calculate errors
        models = {
            'TV-AR': tvpar_forecasts,
            'TV-EWD': tvewd_forecasts,
            'HAR': har_forecasts,
            'TV-HAR': tvhar_forecasts,
            'EWD': ewd_forecasts
        }
        
        for model_name, forecasts in models.items():
            rmse = np.sqrt(mean_squared_error(true_values[h-1:], forecasts[:-h+1] if h > 1 else forecasts))
            mae = mean_absolute_error(true_values[h-1:], forecasts[:-h+1] if h > 1 else forecasts)
            
            results[h][model_name] = {
                'RMSE': rmse,
                'MAE': mae,
                'forecasts': forecasts
            }
    
    # Calculate relative errors compared to TV-HAR
    for h in h_values:
        benchmark_rmse = results[h]['TV-HAR']['RMSE']
        benchmark_mae = results[h]['TV-HAR']['MAE']
        
        print(f"\nResults for {h}-step ahead forecasts:")
        print(f"{'Model':<10} {'RMSE':<10} {'Relative RMSE':<15} {'MAE':<10} {'Relative MAE':<15}")
        print("-" * 60)
        
        for model_name in results[h]:
            rmse = results[h][model_name]['RMSE']
            mae = results[h][model_name]['MAE']
            rel_rmse = rmse / benchmark_rmse
            rel_mae = mae / benchmark_mae
            
            print(f"{model_name:<10} {rmse:<10.4f} {rel_rmse:<15.4f} {mae:<10.4f} {rel_mae:<15.4f}")
    
    # Plot forecasts for each horizon
    for h in h_values:
        plt.figure(figsize=(12, 6))
        plt.plot(true_values, label='True', linewidth=2)
        
        for model_name in ['TV-EWD', 'TV-HAR']:
            forecasts = results[h][model_name]['forecasts']
            plt.plot(forecasts, label=model_name, alpha=0.7)
        
        plt.title(f'{h}-step ahead forecasts')
        plt.xlabel('Time')
        plt.ylabel('Volatility')
        plt.legend()
        plt.show()
    
    # Get persistence structure from the TV-EWD model
    tvewd = TVEWDModel(p=3, J=7, kernel_width=0.3)
    tvewd.fit(data[:window_size])
    persistence = tvewd.get_persistence_structure()
    
    # Plot persistence structure
    plt.figure(figsize=(12, 6))
    scales = [2**j for j in range(1, 8)]
    
    # Create colormap
    cmap = plt.cm.viridis
    norm = plt.Normalize(0, len(scales))
    colors = [cmap(norm(i)) for i in range(len(scales))]
    
    for j in range(persistence.shape[1]):
        plt.plot(persistence[:, j], color=colors[j], label=f'Scale {scales[j]} days')
    
    plt.title('Time-varying persistence structure')
    plt.xlabel('Time')
    plt.ylabel('Relative persistence')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot stacked persistence structure
    plt.figure(figsize=(12, 6))
    
    # Normalize to get relative importance
    plt.stackplot(range(persistence.shape[0]), 
                  [persistence[:, j] for j in range(persistence.shape[1])],
                  labels=[f'Scale {scales[j]} days' for j in range(persistence.shape[1])],
                  colors=colors)
    
    plt.title('Time-varying persistence structure')
    plt.xlabel('Time')
    plt.ylabel('Relative persistence')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    
    return results


def analyze_real_data():
    """
    Load and analyze real energy commodity data
    """
    # Load data (assuming you have historical energy commodity prices)
    # This is a placeholder - in a real application, you would load actual market data
    try:
        # Try to import yfinance for real data
        import yfinance as yf
        
        # Download crude oil futures data
        cl = yf.download("CL=F", start="2010-01-01", end="2022-12-31")
        
        # Calculate daily returns
        cl['returns'] = cl['Close'].pct_change() * 100
        
        # Calculate realized volatility (simple proxy)
        cl['realized_vol'] = cl['returns'].rolling(21).std() * np.sqrt(252)
        
        # Clean and prepare data
        vol_data = cl['realized_vol'].dropna().values
        
        print("Using real crude oil futures data")
    except:
        # If real data is not available, simulate data
        print("Real data not available, using simulated data")
        vol_data = simulate_energy_volatility(n=2000, show_plot=False)
    
    # Analyze persistence structure
    tvewd = TVEWDModel(p=3, J=7, kernel_width=0.3)
    tvewd.fit(vol_data)
    persistence = tvewd.get_persistence_structure()
    
    # Plot persistence structure
    plt.figure(figsize=(12, 6))
    scales = [2**j for j in range(1, 8)]
    
    # Create colormap
    cmap = plt.cm.viridis
    norm = plt.Normalize(0, len(scales))
    colors = [cmap(norm(i)) for i in range(len(scales))]
    
    # Plot stacked persistence structure
    plt.figure(figsize=(12, 6))
    
    # Normalize to get relative importance
    plt.stackplot(range(persistence.shape[0]), 
                  [persistence[:, j] for j in range(persistence.shape[1])],
                  labels=[f'Scale {scales[j]} days' for j in range(persistence.shape[1])],
                  colors=colors)
    
    plt.title('Time-varying persistence structure')
    plt.xlabel('Time')
    plt.ylabel('Relative persistence')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # Calculate which scales are dominant at different times
    dominant_scales = np.argmax(persistence, axis=1)
    
    # Plot dominant scales over time
    plt.figure(figsize=(12, 6))
    plt.plot(dominant_scales)
    plt.yticks(range(7), [f'Scale {scales[j]} days' for j in range(7)])
    plt.title('Dominant persistence scale over time')
    plt.xlabel('Time')
    plt.ylabel('Dominant scale')
    plt.tight_layout()
    plt.show()
    
    return vol_data, persistence


if __name__ == "__main__":
    # Part 1: Simulate energy volatility data
    print("Simulating energy volatility data...")
    volatility = simulate_energy_volatility()
    
    # Part 2: Evaluate models on simulated data
    print("\nEvaluating models on simulated data...")
    results = evaluate_models_simulated_data()
    
    # Part 3: Analyze real data (or simulated if real data not available)
    print("\nAnalyzing persistence structure in real data...")
    vol_data, persistence = analyze_real_data()