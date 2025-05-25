import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_simulated_data(num_assets=100, num_days=1000, seed=42):
    """
    Generate simulated price data for an index and its constituents
    
    Parameters:
    -----------
    num_assets : int
        Number of assets to simulate
    num_days : int
        Number of days to simulate
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    index_data : pandas.Series
        Simulated index returns
    constituent_data : pandas.DataFrame
        Simulated constituent returns
    """
    np.random.seed(seed)
    
    # Create date range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=num_days)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Generate common market factor
    market_factor = np.random.normal(0.0005, 0.01, len(dates))
    
    # Generate asset-specific factors
    asset_betas = np.random.uniform(0.5, 1.5, num_assets)
    asset_specific_risk = np.random.uniform(0.005, 0.02, num_assets)
    
    # Generate constituent returns
    constituent_returns = np.zeros((len(dates), num_assets))
    for i in range(num_assets):
        beta = asset_betas[i]
        specific_risk = asset_specific_risk[i]
        specific_returns = np.random.normal(0, specific_risk, len(dates))
        constituent_returns[:, i] = beta * market_factor + specific_returns
    
    # Create constituent DataFrame
    constituent_columns = [f'Asset_{i+1}' for i in range(num_assets)]
    constituent_data = pd.DataFrame(constituent_returns, index=dates, columns=constituent_columns)
    
    # Generate index returns (weighted average of constituents with some tracking error)
    # Create weights that sum to 1
    true_weights = np.random.uniform(0, 1, num_assets)
    true_weights = true_weights / np.sum(true_weights)
    
    # Make some weights exactly zero to simulate sparse index
    zero_mask = np.random.choice([0, 1], size=num_assets, p=[0.7, 0.3])
    true_weights = true_weights * zero_mask
    if np.sum(true_weights) > 0:
        true_weights = true_weights / np.sum(true_weights)
    
    # Calculate index returns
    index_returns = constituent_data.values @ true_weights
    index_returns = index_returns + np.random.normal(0, 0.001, len(dates))  # Add small tracking error
    index_data = pd.Series(index_returns, index=dates, name='Index')
    
    return index_data, constituent_data, true_weights

def save_simulated_data(filename, num_assets=100, num_days=1000, seed=42):
    """
    Generate and save simulated data to a file
    
    Parameters:
    -----------
    filename : str
        Filename to save the data
    num_assets : int
        Number of assets to simulate
    num_days : int
        Number of days to simulate
    seed : int
        Random seed for reproducibility
    """
    index_data, constituent_data, true_weights = generate_simulated_data(num_assets, num_days, seed)
    
    # Save returns data
    combined_data = pd.concat([index_data, constituent_data], axis=1)
    combined_data.to_csv(filename)
    
    # Save true weights for comparison
    weights_df = pd.DataFrame({
        'Asset': constituent_data.columns,
        'True_Weight': true_weights
    })
    weights_df.to_csv(filename.replace('.csv', '_weights.csv'), index=False)
    
    print(f"Saved simulated data to {filename}")
    print(f"Saved true weights to {filename.replace('.csv', '_weights.csv')}")
    
    return index_data, constituent_data, true_weights

if __name__ == "__main__":
    # Generate and save simulated data
    index_data, constituent_data, true_weights = save_simulated_data(
        'simulated_market_data.csv', 
        num_assets=100, 
        num_days=1000, 
        seed=42
    )
    
    # Print some statistics
    print(f"\nGenerated {len(constituent_data.columns)} assets over {len(constituent_data)} days")
    print(f"Number of active assets in true weights: {np.sum(true_weights > 0)}")
    print(f"Average daily return of index: {index_data.mean():.6f}")
    print(f"Volatility of index: {index_data.std():.6f}")
