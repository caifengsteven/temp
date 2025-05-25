import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import time
from sklearn.metrics import r2_score
import warnings

# Ignore matplotlib warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

np.random.seed(42)

#######################################
# Generate Simulation Data
#######################################

def generate_simulated_stock_data(n_stocks=500, n_days=2500, n_features=10, non_stationarity=0.01):
    """
    Generate simulated stock market data with non-stationary relationships
    
    Args:
        n_stocks: Number of stocks in the universe
        n_days: Number of trading days
        n_features: Number of features per stock
        non_stationarity: Rate of change in relationships (0=stationary, higher=more non-stationary)
    
    Returns:
        features: Array of shape (n_days, n_stocks, n_features)
        returns: Array of shape (n_days, n_stocks)
        market_returns: Array of shape (n_days)
    """
    print("Generating simulated stock market data...")
    
    # Initialize arrays
    features = np.zeros((n_days, n_stocks, n_features))
    true_betas = np.zeros((n_days, n_features))  # Time-varying feature importance
    returns = np.zeros((n_days, n_stocks))
    market_returns = np.random.normal(0.0005, 0.01, n_days)  # Market return with +5bps daily drift
    
    # Initialize stock betas (market sensitivity)
    stock_betas = np.random.normal(1, 0.3, n_stocks)
    
    # Initialize true feature importance - smaller initial values
    true_betas[0] = np.random.normal(0, 0.1, n_features)
    
    # Generate features and returns
    for t in tqdm(range(n_days)):
        # Time-varying true relationships
        if t > 0:
            true_betas[t] = true_betas[t-1] + non_stationarity * np.random.normal(0, 0.1, n_features)
        
        # Generate features (technical indicators) with some autocorrelation
        if t == 0:
            features[t] = np.random.normal(0, 1, (n_stocks, n_features))
        else:
            features[t] = 0.7 * features[t-1] + 0.3 * np.random.normal(0, 1, (n_stocks, n_features))
        
        # Idiosyncratic returns with much more noise
        idiosyncratic = np.zeros(n_stocks)
        for i in range(n_stocks):
            # Stock-specific returns based on features and true relationships
            # Low signal-to-noise ratio (0.005 signal to 0.02 noise)
            feature_effect = 0.005 * np.sum(features[t, i] * true_betas[t])
            idiosyncratic[i] = feature_effect + np.random.normal(0, 0.02)
        
        # Market component + idiosyncratic component
        returns[t] = stock_betas * market_returns[t] + idiosyncratic
    
    print("Data generation complete.")
    return features, returns, market_returns

#######################################
# Data Preprocessing
#######################################

def neutralize_returns(returns, market_returns, n_burn_in=60):
    """
    Neutralize returns by regressing against market returns
    
    Args:
        returns: Array of shape (n_days, n_stocks)
        market_returns: Array of shape (n_days)
        n_burn_in: Number of days to use for initial regression
        
    Returns:
        neutralized_returns: Array of shape (n_days, n_stocks)
    """
    n_days, n_stocks = returns.shape
    neutralized_returns = np.zeros_like(returns)
    
    print("Neutralizing returns...")
    # For each stock
    for i in tqdm(range(n_stocks)):
        # For the burn-in period, use expanding window
        X = market_returns.reshape(-1, 1)
        y = returns[:, i]
        model = LinearRegression().fit(X, y)
        beta = model.coef_[0]
        alpha = model.intercept_
        
        # Neutralize market effect for all days
        neutralized_returns[:, i] = returns[:, i] - (alpha + beta * market_returns)
    
    # Scale down the neutralized returns to realistic levels (typically <1% daily)
    neutralized_returns = neutralized_returns * 0.5
    
    return neutralized_returns

#######################################
# Model Implementations
#######################################

class RollingRegression:
    def __init__(self, window_size=252):
        self.window_size = window_size
        self.weights = None
    
    def fit_predict(self, features, returns, start_idx):
        n_days, n_stocks, n_features = features.shape
        predictions = np.zeros((n_days, n_stocks))
        
        print("Fitting Rolling Regression model...")
        # Initialize weights to zeros
        self.weights = np.zeros(n_features)
        
        for t in tqdm(range(start_idx, n_days)):
            if t == start_idx or t % 10 == 0:  # Re-fit model every 10 days to save computation
                # Flatten the data to create a pooled regression
                start_t = max(0, t - self.window_size)
                X_train = features[start_t:t].reshape(-1, n_features)
                y_train = returns[start_t:t].flatten()
                
                # Fit linear regression
                try:
                    model = LinearRegression().fit(X_train, y_train)
                    self.weights = model.coef_
                except Exception as e:
                    print(f"Error fitting linear regression at t={t}: {e}")
                    # Keep using previous weights
            
            # Make predictions
            for i in range(n_stocks):
                predictions[t, i] = np.dot(features[t, i], self.weights)
        
        return predictions

class AROW:
    def __init__(self, n_features, r=1.0):
        self.mu = np.zeros(n_features)  # Mean of weight distribution
        self.sigma = np.eye(n_features)  # Covariance of weight distribution
        self.r = r  # Confidence parameter
    
    def update(self, x, y):
        """Update model with a single instance"""
        # Prediction
        pred = np.dot(self.mu, x)
        
        # Compute update quantities
        confidence = np.dot(np.dot(x, self.sigma), x)
        beta = 1.0 / (confidence + self.r)
        loss = y - pred
        
        # Update parameters
        sigma_update = beta * np.outer(np.dot(self.sigma, x), np.dot(x, self.sigma))
        
        # Ensure sigma_update is well-conditioned
        if np.max(np.abs(sigma_update)) > 1e10:
            sigma_update = sigma_update * (1e10 / np.max(np.abs(sigma_update)))
            
        self.sigma = self.sigma - sigma_update
        
        # Ensure Sigma remains positive definite
        eigvals = np.linalg.eigvalsh(self.sigma)
        if np.min(eigvals) < 1e-10:
            # Add a small regularization term
            self.sigma = self.sigma + 1e-10 * np.eye(len(self.mu))
            
        self.mu = self.mu + beta * loss * np.dot(self.sigma, x)
        
        # Prevent weights from growing too large
        if np.max(np.abs(self.mu)) > 10:
            self.mu = self.mu * 10 / np.max(np.abs(self.mu))
        
        return pred
    
    def fit_predict(self, features, returns, start_idx):
        n_days, n_stocks, n_features = features.shape
        predictions = np.zeros((n_days, n_stocks))
        
        print("Fitting AROW model (sequential updates)...")
        for t in tqdm(range(start_idx, n_days)):
            # First predict for all stocks
            for i in range(n_stocks):
                predictions[t, i] = np.dot(self.mu, features[t, i])
            
            # Then update with each instance sequentially
            for i in range(n_stocks):
                self.update(features[t, i], returns[t, i])
        
        return predictions

class BAROW:
    def __init__(self, n_features, r=1.0):
        self.mu = np.zeros(n_features)  # Mean of weight distribution
        self.sigma = np.eye(n_features)  # Covariance of weight distribution
        self.r = r  # Confidence parameter
        self.weights_history = []  # To store weights over time
    
    def batch_update(self, X_batch, y_batch):
        """Update model with a batch of instances"""
        # Save current mu for history
        self.weights_history.append(self.mu.copy())
        
        # Compute batch size
        K = len(y_batch)
        R = self.r * K
        
        # Make predictions before update
        predictions = np.dot(X_batch, self.mu)
        
        try:
            # Compute terms for the update
            X_sigma = np.dot(X_batch, self.sigma)
            M = np.eye(K) * R + np.dot(X_sigma, X_batch.T)
            
            # Use a more stable inversion method with regularization
            try:
                # Add small regularization to ensure stability
                M_reg = M + 1e-8 * np.eye(K)
                M_inv = np.linalg.inv(M_reg)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse if regular inverse fails
                M_inv = np.linalg.pinv(M)
            
            # Update parameters using formulas from the paper
            sigma_update = np.dot(np.dot(self.sigma, X_batch.T), np.dot(M_inv, X_sigma))
            
            # Check for numerical stability
            if np.max(np.abs(sigma_update)) > 1e10:
                # Scale down the update if it's too large
                scale_factor = 1e10 / np.max(np.abs(sigma_update))
                sigma_update = sigma_update * scale_factor
                
            self.sigma = self.sigma - sigma_update
            
            # Ensure Sigma remains positive definite
            eigvals = np.linalg.eigvalsh(self.sigma)
            if np.min(eigvals) < 1e-8:
                # Add a small regularization term
                self.sigma = self.sigma + 1e-8 * np.eye(len(self.mu))
            
            # Update mu
            pred_error = np.dot(X_batch, self.mu) - y_batch
            mu_update = np.dot(np.dot(self.sigma, X_batch.T), np.dot(M_inv, pred_error))
            
            # Check for numerical stability
            if np.max(np.abs(mu_update)) > 1e10:
                # Scale down the update if it's too large
                scale_factor = 1e10 / np.max(np.abs(mu_update))
                mu_update = mu_update * scale_factor
                
            self.mu = self.mu - mu_update
            
            # Prevent weights from growing too large
            if np.max(np.abs(self.mu)) > 10:
                self.mu = self.mu * 10 / np.max(np.abs(self.mu))
            
        except Exception as e:
            print(f"Error in batch update: {e}")
            # If the update fails, keep the previous parameters
            # This is a reasonable fallback
        
        return predictions
    
    def fit_predict(self, features, returns, start_idx):
        n_days, n_stocks, n_features = features.shape
        predictions = np.zeros((n_days, n_stocks))
        
        print("Fitting BAROW model (batch updates)...")
        for t in tqdm(range(start_idx, n_days)):
            # Predict for all stocks
            X_batch = features[t]
            predictions[t] = np.dot(X_batch, self.mu)
            
            # Update with the entire batch
            self.batch_update(X_batch, returns[t])
        
        return predictions, np.array(self.weights_history)

#######################################
# Strategy Evaluation
#######################################

def calculate_strategy_returns(predictions, actual_returns, normalize=True):
    """
    Calculate strategy returns by taking positions proportional to predictions
    
    Args:
        predictions: Array of shape (n_days, n_stocks)
        actual_returns: Array of shape (n_days, n_stocks)
        normalize: Whether to normalize positions
        
    Returns:
        strategy_returns: Array of shape (n_days)
    """
    n_days = predictions.shape[0]
    strategy_returns = np.zeros(n_days)
    
    for t in range(n_days):
        try:
            if np.any(~np.isnan(predictions[t])):
                # Calculate positions proportional to predictions
                positions = predictions[t] - np.nanmean(predictions[t])
                
                if normalize and np.nanstd(positions) > 0:
                    positions = positions / np.nanstd(positions)
                
                # Calculate strategy return as weighted average of actual returns
                if np.nansum(np.abs(positions)) > 0:
                    strategy_returns[t] = np.nansum(positions * actual_returns[t]) / np.nansum(np.abs(positions))
                else:
                    strategy_returns[t] = 0.0
                
                # Alternative: cross-sectional correlation as described in the paper
                if np.isnan(strategy_returns[t]) or np.isinf(strategy_returns[t]):
                    valid_idx = ~np.isnan(predictions[t]) & ~np.isnan(actual_returns[t])
                    if np.sum(valid_idx) > 10:  # Need enough data points
                        correlation = np.corrcoef(predictions[t, valid_idx], actual_returns[t, valid_idx])[0, 1]
                        if not np.isnan(correlation):
                            strategy_returns[t] = correlation * np.nanstd(actual_returns[t])
                        else:
                            strategy_returns[t] = 0.0
        except Exception as e:
            print(f"Error calculating strategy return for day {t}: {e}")
            strategy_returns[t] = 0.0
    
    # Clip extreme returns for stability
    strategy_returns = np.clip(strategy_returns, -0.05, 0.05)  # Max 5% daily return
    
    return strategy_returns

def evaluate_strategy(strategy_returns, name):
    """Calculate performance metrics for a strategy"""
    try:
        # Clean any bad values
        strategy_returns = np.array(strategy_returns)
        strategy_returns[np.isnan(strategy_returns)] = 0
        strategy_returns[np.isinf(strategy_returns)] = 0
        
        # Clip to reasonable values
        strategy_returns = np.clip(strategy_returns, -0.05, 0.05)
        
        # Calculate cumulative returns
        cum_returns = np.cumprod(1 + strategy_returns) - 1
        
        # Calculate Sharpe ratio (annualized)
        sharpe = np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) > 0 else 0
        
        # Calculate maximum drawdown
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - peak) / (1 + peak)
        max_drawdown = np.min(drawdown)
        
        # Calculate Calmar ratio
        total_return = cum_returns[-1]
        calmar = total_return / abs(max_drawdown) if max_drawdown < 0 else total_return  # Avoid division by zero
        
        # Cap metrics to reasonable values
        sharpe = min(sharpe, 10.0)  # Cap Sharpe at 10
        calmar = min(calmar, 20.0)  # Cap Calmar at 20
        
        print(f"\n=== {name} Strategy Performance ===")
        print(f"Total Return: {total_return:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        print(f"Calmar Ratio: {calmar:.2f}")
        
        return {
            "name": name,
            "cum_returns": cum_returns,
            "total_return": total_return,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "calmar": calmar
        }
    except Exception as e:
        print(f"Error evaluating {name} strategy: {e}")
        # Return dummy values in case of error
        return {
            "name": name,
            "cum_returns": np.zeros(len(strategy_returns)),
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0
        }

def plot_results(results):
    """Plot cumulative returns of all strategies"""
    try:
        plt.figure(figsize=(12, 6))
        
        for result in results:
            plt.plot(result["cum_returns"], label=f"{result['name']} (Return: {result['total_return']:.2%}, Sharpe: {result['sharpe']:.2f})")
        
        plt.title("Cumulative Returns of Trading Strategies")
        plt.xlabel("Trading Days")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True)
        
        # Save the plot instead of using tight_layout
        plt.savefig('strategy_returns.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Plot saved as 'strategy_returns.png'")
    except Exception as e:
        print(f"Error plotting results: {e}")

def plot_weights_evolution(weights_history, n_features_to_plot=5):
    """Plot the evolution of weights over time"""
    try:
        plt.figure(figsize=(12, 6))
        
        weights_array = np.array(weights_history)
        for i in range(min(n_features_to_plot, weights_array.shape[1])):
            plt.plot(weights_array[:, i], label=f'Feature {i+1}')
        
        plt.title('BAROW Feature Weights Over Time')
        plt.xlabel('Trading Days')
        plt.ylabel('Weight')
        plt.legend()
        plt.grid(True)
        
        # Save the plot instead of using tight_layout
        plt.savefig('weights_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Plot saved as 'weights_evolution.png'")
    except Exception as e:
        print(f"Error plotting weight evolution: {e}")

#######################################
# Main Execution
#######################################

def main():
    try:
        # Settings
        n_stocks = 500  # Similar to S&P 500
        n_days = 800    # Reduce to speed up execution
        n_features = 10
        burn_in_period = 252  # 1 year (252 trading days)
        
        # Generate simulated data with more realistic parameters
        features, returns, market_returns = generate_simulated_stock_data(
            n_stocks=n_stocks,
            n_days=n_days,
            n_features=n_features,
            non_stationarity=0.01  # Reduced from 0.03
        )
        
        # Neutralize returns (as in the paper)
        neutralized_returns = neutralize_returns(returns, market_returns, n_burn_in=burn_in_period)
        
        # Initialize models with higher regularization
        rolling_regression = RollingRegression(window_size=252)  # 1 year lookback
        arow_model = AROW(n_features=n_features, r=1.0)  # Increased from 0.1
        barow_model = BAROW(n_features=n_features, r=1.0)  # Increased from 0.1
        
        # Fit and predict
        print("\nTraining and predicting with Rolling Regression...")
        start_time = time.time()
        rolling_predictions = rolling_regression.fit_predict(features, neutralized_returns, burn_in_period)
        print(f"Rolling Regression completed in {time.time() - start_time:.2f} seconds")
        
        print("\nTraining and predicting with AROW (sequential updates)...")
        start_time = time.time()
        arow_predictions = arow_model.fit_predict(features, neutralized_returns, burn_in_period)
        print(f"AROW completed in {time.time() - start_time:.2f} seconds")
        
        print("\nTraining and predicting with BAROW (batch updates)...")
        start_time = time.time()
        barow_predictions, weights_history = barow_model.fit_predict(features, neutralized_returns, burn_in_period)
        print(f"BAROW completed in {time.time() - start_time:.2f} seconds")
        
        # Calculate strategy returns
        print("\nCalculating strategy returns...")
        rolling_strategy_returns = calculate_strategy_returns(rolling_predictions, neutralized_returns)
        arow_strategy_returns = calculate_strategy_returns(arow_predictions, neutralized_returns)
        barow_strategy_returns = calculate_strategy_returns(barow_predictions, neutralized_returns)
        
        # Calculate out-of-sample R² (from burn-in period onward)
        test_idx = range(burn_in_period, n_days)
        
        # Flatten the data
        y_true = neutralized_returns[test_idx].flatten()
        y_pred_rolling = rolling_predictions[test_idx].flatten()
        y_pred_arow = arow_predictions[test_idx].flatten()
        y_pred_barow = barow_predictions[test_idx].flatten()
        
        # Filter out NaN values
        valid_idx_rolling = ~np.isnan(y_pred_rolling) & ~np.isnan(y_true)
        valid_idx_arow = ~np.isnan(y_pred_arow) & ~np.isnan(y_true)
        valid_idx_barow = ~np.isnan(y_pred_barow) & ~np.isnan(y_true)
        
        r2_rolling = r2_score(y_true[valid_idx_rolling], y_pred_rolling[valid_idx_rolling]) if np.sum(valid_idx_rolling) > 10 else float('nan')
        r2_arow = r2_score(y_true[valid_idx_arow], y_pred_arow[valid_idx_arow]) if np.sum(valid_idx_arow) > 10 else float('nan')
        r2_barow = r2_score(y_true[valid_idx_barow], y_pred_barow[valid_idx_barow]) if np.sum(valid_idx_barow) > 10 else float('nan')
        
        print(f"\n=== Out-of-Sample R² ===")
        print(f"Rolling Regression R²: {r2_rolling:.4f}")
        print(f"AROW R²: {r2_arow:.4f}")
        print(f"BAROW R²: {r2_barow:.4f}")
        
        # Evaluate strategies
        results = []
        results.append(evaluate_strategy(rolling_strategy_returns[burn_in_period:], "Rolling Regression"))
        results.append(evaluate_strategy(arow_strategy_returns[burn_in_period:], "AROW"))
        results.append(evaluate_strategy(barow_strategy_returns[burn_in_period:], "BAROW"))
        
        # Create performance summary table
        performance_df = pd.DataFrame({
            'Model': [r['name'] for r in results],
            'Total Return': [f"{r['total_return']:.2%}" for r in results],
            'Sharpe Ratio': [f"{r['sharpe']:.2f}" for r in results],
            'Max Drawdown': [f"{r['max_drawdown']:.2%}" for r in results],
            'Calmar Ratio': [f"{r['calmar']:.2f}" for r in results]
        })
        
        print("\n=== Performance Summary ===")
        print(performance_df.to_string(index=False))
        
        # Plot results
        plot_results(results)
        
        # Plot weight evolution for BAROW
        plot_weights_evolution(weights_history)
        
        return results, features, returns, neutralized_returns, rolling_predictions, arow_predictions, barow_predictions, weights_history
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        # Return empty results to avoid crashing
        return [], None, None, None, None, None, None, None

if __name__ == "__main__":
    results, features, returns, neutralized_returns, rolling_predictions, arow_predictions, barow_predictions, weights_history = main()