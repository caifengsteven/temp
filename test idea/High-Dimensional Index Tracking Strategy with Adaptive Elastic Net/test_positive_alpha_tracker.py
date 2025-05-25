import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Import the simulation and tracker code
from simulate_data import generate_simulated_data
from EnhancedIndexTracker_positive_alpha import EnhancedIndexTrackerPositiveAlpha

# Set plotting style
plt.style.use('seaborn-whitegrid')
sns.set_context("paper", font_scale=1.5)

# For reproducibility
np.random.seed(42)

def test_positive_alpha_tracker(num_assets=100, num_days=1000, 
                              lambda1=1e-5, lambda2=1e-2, lambda_c=1e-4, tau=1,
                              index_weight=0.6, active_weight=0.4,
                              lookback_window=250, rebalance_period=21):
    """
    Test the Enhanced Index Tracker with Positive Alpha Focus
    
    Parameters:
    -----------
    num_assets : int
        Number of assets to simulate
    num_days : int
        Number of days to simulate
    lambda1, lambda2, lambda_c, tau : float
        Parameters for the adaptive elastic net model
    index_weight : float
        Weight for the index-tracking component
    active_weight : float
        Weight for the active component
    lookback_window : int
        Number of days to use for training
    rebalance_period : int
        Number of days between rebalances
    """
    print("Generating simulated data...")
    index_data, constituent_data, true_weights = generate_simulated_data(num_assets, num_days)
    
    # Split data into training and testing periods
    train_size = lookback_window
    
    # Initialize performance tracking
    performance = {
        'date': [],
        'portfolio_return': [],
        'index_return': [],
        'tracking_error': [],
        'active_return': [],
        'num_assets': [],
        'turnover': [],
        'alpha_exposure': [],
        'beta_exposure': [],
        'expected_return': [],
        'active_weight': []
    }
    
    # Initialize model
    model = EnhancedIndexTrackerPositiveAlpha(
        lambda1=lambda1, 
        lambda2=lambda2, 
        lambda_c=lambda_c, 
        tau=tau,
        index_weight=index_weight,
        active_weight=active_weight
    )
    
    # Initialize weights
    current_weights = None
    
    # Main backtest loop
    print("\nRunning backtest...")
    for i in range(train_size, len(index_data), rebalance_period):
        # Determine training window
        train_start_idx = max(0, i - lookback_window)
        train_end_idx = i
        
        # Prepare training data
        X_train = constituent_data.iloc[train_start_idx:train_end_idx].values
        y_train = index_data.iloc[train_start_idx:train_end_idx].values
        
        # Fit model
        start_time = time.time()
        model.fit(X_train, y_train, current_weights)
        end_time = time.time()
        
        print(f"Fitting model took {end_time - start_time:.2f} seconds")
        
        # Get weights and active assets
        new_weights = model.get_weights()
        active_assets = model.get_active_assets()
        
        # Get factor exposures
        factor_exposures = model.get_factor_exposures()
        
        # Calculate turnover
        turnover = 0
        if current_weights is not None:
            turnover = np.sum(np.abs(new_weights - current_weights))
        
        # Update current weights
        current_weights = new_weights
        
        # Determine out-of-sample period
        test_start_idx = i
        test_end_idx = min(i + rebalance_period, len(index_data))
        
        # Calculate out-of-sample returns
        for j in range(test_start_idx, test_end_idx):
            if j >= len(index_data):
                continue
                
            date = index_data.index[j]
            
            # Get constituent returns for this day
            if j > 0:  # Skip first day (no return calculation possible)
                const_returns = constituent_data.iloc[j]
                index_return = index_data.iloc[j]
                
                # Calculate portfolio return
                portfolio_return = np.sum(const_returns * current_weights)
                
                # Calculate tracking error and active return
                tracking_error = (portfolio_return - index_return) ** 2
                active_return = portfolio_return - index_return
                
                # Store performance
                performance['date'].append(date)
                performance['portfolio_return'].append(portfolio_return)
                performance['index_return'].append(index_return)
                performance['tracking_error'].append(tracking_error)
                performance['active_return'].append(active_return)
                performance['num_assets'].append(len(active_assets))
                performance['turnover'].append(turnover)
                
                # Store factor exposures
                if factor_exposures:
                    performance['alpha_exposure'].append(factor_exposures['alpha'])
                    performance['beta_exposure'].append(factor_exposures['beta'])
                    performance['expected_return'].append(factor_exposures['expected_return'])
                    performance['active_weight'].append(factor_exposures['active_weight'])
                else:
                    performance['alpha_exposure'].append(0)
                    performance['beta_exposure'].append(1.0)
                    performance['expected_return'].append(0)
                    performance['active_weight'].append(0)
        
        # Print progress
        print(f"Completed period ending {index_data.index[min(test_end_idx-1, len(index_data)-1)].strftime('%Y-%m-%d')}")
        print(f"Number of active assets: {len(active_assets)}")
        print(f"Turnover: {turnover:.4f}")
        if factor_exposures:
            print(f"Factor exposures: Alpha={factor_exposures['alpha']:.6f}, "
                  f"Beta={factor_exposures['beta']:.4f}, "
                  f"Expected Return={factor_exposures['expected_return']:.6f}, "
                  f"Active Weight={factor_exposures['active_weight']:.4f}")
        print("-" * 50)
    
    # Convert performance to DataFrame
    tracking_performance = pd.DataFrame(performance)
    
    # Calculate cumulative performance
    tracking_performance['cumulative_portfolio'] = (1 + tracking_performance['portfolio_return']).cumprod()
    tracking_performance['cumulative_index'] = (1 + tracking_performance['index_return']).cumprod()
    
    # Calculate and print metrics
    calculate_performance_metrics(tracking_performance)
    
    # Compare estimated weights with true weights
    compare_weights(current_weights, true_weights, constituent_data.columns)
    
    # Plot performance
    plot_performance(tracking_performance)
    
    return tracking_performance, current_weights

def calculate_performance_metrics(tracking_performance):
    """Calculate and print summary performance metrics"""
    if tracking_performance is None or len(tracking_performance) == 0:
        print("No performance data available")
        return
    
    # Calculate metrics
    annualized_tracking_error = np.sqrt(np.mean(tracking_performance['tracking_error'])) * np.sqrt(252)
    annualized_active_return = np.mean(tracking_performance['active_return']) * 252
    annualized_portfolio_return = np.mean(tracking_performance['portfolio_return']) * 252
    annualized_index_return = np.mean(tracking_performance['index_return']) * 252
    
    correlation = np.corrcoef(
        tracking_performance['portfolio_return'],
        tracking_performance['index_return']
    )[0, 1]
    
    average_num_assets = np.mean(tracking_performance['num_assets'])
    average_turnover = np.mean(tracking_performance['turnover'])
    
    # Calculate information ratio
    information_ratio = annualized_active_return / annualized_tracking_error if annualized_tracking_error > 0 else 0
    
    # Calculate average factor exposures
    avg_alpha_exposure = np.mean(tracking_performance['alpha_exposure'])
    avg_beta_exposure = np.mean(tracking_performance['beta_exposure'])
    avg_expected_return = np.mean(tracking_performance['expected_return'])
    avg_active_weight = np.mean(tracking_performance['active_weight'])
    
    # Print metrics
    print("\nPerformance Metrics:")
    print(f"Annualized Tracking Error: {annualized_tracking_error:.4f}")
    print(f"Annualized Active Return: {annualized_active_return:.4f}")
    print(f"Annualized Portfolio Return: {annualized_portfolio_return:.4f}")
    print(f"Annualized Index Return: {annualized_index_return:.4f}")
    print(f"Information Ratio: {information_ratio:.4f}")
    print(f"Correlation with Index: {correlation:.4f}")
    print(f"Average Number of Active Assets: {average_num_assets:.2f}")
    print(f"Average Turnover: {average_turnover:.4f}")
    print(f"Average Alpha Exposure: {avg_alpha_exposure:.6f}")
    print(f"Average Beta Exposure: {avg_beta_exposure:.4f}")
    print(f"Average Expected Return: {avg_expected_return:.6f}")
    print(f"Average Active Weight: {avg_active_weight:.4f}")

def compare_weights(estimated_weights, true_weights, asset_names):
    """Compare estimated weights with true weights"""
    # Create DataFrame for comparison
    weight_comparison = pd.DataFrame({
        'Asset': asset_names,
        'Estimated_Weight': estimated_weights,
        'True_Weight': true_weights
    })
    
    # Sort by true weight
    weight_comparison = weight_comparison.sort_values('True_Weight', ascending=False)
    
    # Calculate metrics
    correlation = np.corrcoef(estimated_weights, true_weights)[0, 1]
    mse = np.mean((estimated_weights - true_weights) ** 2)
    mae = np.mean(np.abs(estimated_weights - true_weights))
    
    # Print metrics
    print("\nWeight Comparison Metrics:")
    print(f"Correlation between estimated and true weights: {correlation:.4f}")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"Number of active assets in true weights: {np.sum(true_weights > 0)}")
    print(f"Number of active assets in estimated weights: {np.sum(estimated_weights > 1e-6)}")
    
    # Save comparison to CSV
    weight_comparison.to_csv('positive_alpha_weight_comparison.csv', index=False)
    print("Saved weight comparison to positive_alpha_weight_comparison.csv")
    
    # Plot top assets
    plt.figure(figsize=(12, 8))
    top_n = min(20, len(asset_names))
    top_assets = weight_comparison.head(top_n)
    
    x = np.arange(len(top_assets))
    width = 0.35
    
    plt.bar(x - width/2, top_assets['True_Weight'], width, label='True Weights')
    plt.bar(x + width/2, top_assets['Estimated_Weight'], width, label='Estimated Weights')
    
    plt.xlabel('Assets')
    plt.ylabel('Weight')
    plt.title('Top Assets: True vs Estimated Weights')
    plt.xticks(x, top_assets['Asset'], rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig('positive_alpha_weight_comparison.png')
    plt.show()

def plot_performance(tracking_performance):
    """Plot performance of the tracking portfolio"""
    if tracking_performance is None or len(tracking_performance) == 0:
        print("No performance data available")
        return
    
    fig, axs = plt.subplots(5, 1, figsize=(12, 25), gridspec_kw={'height_ratios': [3, 1, 1, 1, 1]})
    
    # Plot cumulative returns
    axs[0].plot(tracking_performance['date'], tracking_performance['cumulative_portfolio'], 
               label='Positive Alpha Portfolio', linewidth=2)
    axs[0].plot(tracking_performance['date'], tracking_performance['cumulative_index'], 
               label='Index', linewidth=2, linestyle='--')
    axs[0].set_title('Cumulative Performance')
    axs[0].set_ylabel('Value')
    axs[0].set_xlabel('Date')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot active returns
    axs[1].plot(tracking_performance['date'], tracking_performance['active_return'], 
               color='green', label='Active Return')
    axs[1].axhline(y=0, color='r', linestyle='-')
    axs[1].set_title('Active Returns')
    axs[1].set_ylabel('Return')
    axs[1].set_xlabel('Date')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot number of active assets
    axs[2].plot(tracking_performance['date'], tracking_performance['num_assets'], 
               color='purple', label='Active Assets')
    axs[2].set_title('Number of Active Assets')
    axs[2].set_ylabel('Count')
    axs[2].set_xlabel('Date')
    axs[2].legend()
    axs[2].grid(True)
    
    # Plot alpha and beta exposures
    axs[3].plot(tracking_performance['date'], tracking_performance['alpha_exposure'], 
               color='blue', label='Alpha Exposure')
    axs[3].plot(tracking_performance['date'], tracking_performance['beta_exposure'], 
               color='orange', label='Beta Exposure')
    axs[3].axhline(y=0, color='r', linestyle='-')
    axs[3].axhline(y=1, color='r', linestyle='--')
    axs[3].set_title('Alpha and Beta Exposures')
    axs[3].set_ylabel('Exposure')
    axs[3].set_xlabel('Date')
    axs[3].legend()
    axs[3].grid(True)
    
    # Plot expected return and active weight
    axs[4].plot(tracking_performance['date'], tracking_performance['expected_return'], 
               color='green', label='Expected Return')
    axs[4].plot(tracking_performance['date'], tracking_performance['active_weight'], 
               color='red', label='Active Weight')
    axs[4].set_title('Expected Return and Active Weight')
    axs[4].set_ylabel('Value')
    axs[4].set_xlabel('Date')
    axs[4].legend()
    axs[4].grid(True)
    
    plt.tight_layout()
    plt.savefig('positive_alpha_performance.png')
    plt.show()

if __name__ == "__main__":
    # Test with simulated data
    test_positive_alpha_tracker(
        num_assets=100,
        num_days=1000,
        lambda1=1e-5,
        lambda2=1e-2,
        lambda_c=1e-4,
        tau=1,
        index_weight=0.6,
        active_weight=0.4,
        lookback_window=250,
        rebalance_period=21
    )
