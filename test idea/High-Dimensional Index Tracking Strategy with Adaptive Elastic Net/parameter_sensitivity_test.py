import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Import the simulation and strategy code
from simulate_data import generate_simulated_data
from test_strategy_with_simulated_data import test_with_simulated_data

# Set plotting style
plt.style.use('seaborn-whitegrid')
sns.set_context("paper", font_scale=1.5)

# For reproducibility
np.random.seed(42)

def run_parameter_sensitivity_test():
    """
    Run tests with different parameter settings to analyze sensitivity
    """
    # Create results directory
    results_dir = "parameter_test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate one set of simulated data to use for all tests
    print("Generating simulated data...")
    index_data, constituent_data, true_weights = generate_simulated_data(
        num_assets=100, 
        num_days=1000
    )
    
    # Save the data for reference
    combined_data = pd.concat([index_data, constituent_data], axis=1)
    combined_data.to_csv(f"{results_dir}/simulated_data.csv")
    
    # Define parameter ranges to test
    lambda1_values = [1e-6, 1e-5, 1e-4]
    lambda2_values = [1e-4, 1e-3, 1e-2]
    lambda_c_values = [0, 1e-5, 1e-4]
    
    # Initialize results storage
    results = []
    
    # Run tests for each parameter combination
    total_tests = len(lambda1_values) * len(lambda2_values) * len(lambda_c_values)
    test_count = 0
    
    for lambda1 in lambda1_values:
        for lambda2 in lambda2_values:
            for lambda_c in lambda_c_values:
                test_count += 1
                print(f"\nRunning test {test_count}/{total_tests}")
                print(f"Parameters: lambda1={lambda1}, lambda2={lambda2}, lambda_c={lambda_c}")
                
                # Create a test ID
                test_id = f"l1_{lambda1}_l2_{lambda2}_lc_{lambda_c}"
                
                # Run the test
                start_time = time.time()
                
                # Use the test function but with our pre-generated data
                from test_strategy_with_simulated_data import AdaptiveElasticNetIndexTracker
                
                # Initialize model with current parameters
                model = AdaptiveElasticNetIndexTracker(lambda1, lambda2, lambda_c, tau=1)
                
                # Split data
                train_size = 250  # lookback window
                
                # Prepare training data for initial fit
                X_train = constituent_data.iloc[:train_size].values
                y_train = index_data.iloc[:train_size].values
                
                # Fit model
                model.fit(X_train, y_train)
                
                # Get weights and active assets
                weights = model.get_weights()
                active_assets = model.get_active_assets()
                tracking_error = model.get_tracking_error()
                
                # Calculate weight comparison metrics
                correlation = np.corrcoef(weights, true_weights)[0, 1]
                mse = np.mean((weights - true_weights) ** 2)
                mae = np.mean(np.abs(weights - true_weights))
                
                end_time = time.time()
                
                # Store results
                result = {
                    'test_id': test_id,
                    'lambda1': lambda1,
                    'lambda2': lambda2,
                    'lambda_c': lambda_c,
                    'num_active_assets': len(active_assets),
                    'tracking_error': tracking_error,
                    'weight_correlation': correlation,
                    'weight_mse': mse,
                    'weight_mae': mae,
                    'runtime': end_time - start_time
                }
                
                results.append(result)
                
                # Print key metrics
                print(f"Number of active assets: {len(active_assets)}")
                print(f"In-sample tracking error: {tracking_error:.6f}")
                print(f"Weight correlation: {correlation:.4f}")
                print(f"Weight MSE: {mse:.6f}")
                print(f"Runtime: {end_time - start_time:.2f} seconds")
                
                # Save weights for this test
                weight_df = pd.DataFrame({
                    'Asset': constituent_data.columns,
                    'Estimated_Weight': weights,
                    'True_Weight': true_weights
                })
                weight_df.to_csv(f"{results_dir}/weights_{test_id}.csv", index=False)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(f"{results_dir}/parameter_test_results.csv", index=False)
    
    # Create summary plots
    create_parameter_sensitivity_plots(results_df, results_dir)
    
    return results_df

def create_parameter_sensitivity_plots(results_df, results_dir):
    """Create plots to visualize parameter sensitivity"""
    # Plot number of active assets vs parameters
    plt.figure(figsize=(15, 10))
    
    # Create a pivot table for heatmap
    for i, lambda_c in enumerate(results_df['lambda_c'].unique()):
        subset = results_df[results_df['lambda_c'] == lambda_c]
        pivot = subset.pivot(index='lambda1', columns='lambda2', values='num_active_assets')
        
        plt.subplot(1, 3, i+1)
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu")
        plt.title(f'Number of Active Assets (lambda_c={lambda_c})')
        plt.xlabel('lambda2')
        plt.ylabel('lambda1')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/active_assets_heatmap.png")
    
    # Plot tracking error vs parameters
    plt.figure(figsize=(15, 10))
    
    for i, lambda_c in enumerate(results_df['lambda_c'].unique()):
        subset = results_df[results_df['lambda_c'] == lambda_c]
        pivot = subset.pivot(index='lambda1', columns='lambda2', values='tracking_error')
        
        plt.subplot(1, 3, i+1)
        sns.heatmap(pivot, annot=True, fmt=".6f", cmap="YlOrRd")
        plt.title(f'Tracking Error (lambda_c={lambda_c})')
        plt.xlabel('lambda2')
        plt.ylabel('lambda1')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/tracking_error_heatmap.png")
    
    # Plot weight correlation vs parameters
    plt.figure(figsize=(15, 10))
    
    for i, lambda_c in enumerate(results_df['lambda_c'].unique()):
        subset = results_df[results_df['lambda_c'] == lambda_c]
        pivot = subset.pivot(index='lambda1', columns='lambda2', values='weight_correlation')
        
        plt.subplot(1, 3, i+1)
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap="Greens")
        plt.title(f'Weight Correlation (lambda_c={lambda_c})')
        plt.xlabel('lambda2')
        plt.ylabel('lambda1')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/weight_correlation_heatmap.png")
    
    # Create a summary plot
    plt.figure(figsize=(10, 6))
    
    # Sort by tracking error
    sorted_results = results_df.sort_values('tracking_error')
    
    # Plot top 5 parameter combinations
    top_5 = sorted_results.head(5)
    
    x = np.arange(len(top_5))
    width = 0.2
    
    plt.bar(x - width, top_5['tracking_error'], width, label='Tracking Error')
    plt.bar(x, top_5['weight_correlation'], width, label='Weight Correlation')
    plt.bar(x + width, top_5['num_active_assets'] / 100, width, label='Active Assets %')
    
    plt.xlabel('Parameter Combination')
    plt.ylabel('Value')
    plt.title('Top 5 Parameter Combinations')
    plt.xticks(x, [f"l1={row['lambda1']}, l2={row['lambda2']}, lc={row['lambda_c']}" 
                  for _, row in top_5.iterrows()], rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/top_5_parameters.png")
    
    # Print summary of best parameters
    print("\nTop 5 Parameter Combinations by Tracking Error:")
    for i, (_, row) in enumerate(top_5.iterrows()):
        print(f"{i+1}. lambda1={row['lambda1']}, lambda2={row['lambda2']}, lambda_c={row['lambda_c']}")
        print(f"   Tracking Error: {row['tracking_error']:.6f}")
        print(f"   Weight Correlation: {row['weight_correlation']:.4f}")
        print(f"   Active Assets: {row['num_active_assets']}")
        print(f"   Weight MSE: {row['weight_mse']:.6f}")
        print()

if __name__ == "__main__":
    # Run parameter sensitivity test
    results = run_parameter_sensitivity_test()
