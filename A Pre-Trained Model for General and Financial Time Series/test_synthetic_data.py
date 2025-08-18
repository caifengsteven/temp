"""
Test script for synthetic data generation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from delphyne.data import WaveletDataGenerator, GARCHDataGenerator, SyntheticDataset
from delphyne import DelphyneModel, DelphyneConfig
from torch.utils.data import DataLoader

def test_wavelet_generation():
    """Test Wavelet data generation."""
    print("Testing Wavelet data generation...")
    
    generator = WaveletDataGenerator(seed=42)
    
    # Test univariate
    data, metadata = generator.generate(batch_size=5, seq_len=100, num_variates=1)
    print(f"Univariate Wavelet shape: {data.shape}")
    print(f"Metadata: {metadata['type']}")
    
    # Test multivariate correlated
    data_corr, _ = generator.generate(batch_size=5, seq_len=100, num_variates=3, correlated=True)
    print(f"Multivariate correlated Wavelet shape: {data_corr.shape}")
    
    # Test multivariate uncorrelated
    data_uncorr, _ = generator.generate(batch_size=5, seq_len=100, num_variates=3, correlated=False)
    print(f"Multivariate uncorrelated Wavelet shape: {data_uncorr.shape}")
    
    return True

def test_garch_generation():
    """Test GARCH data generation."""
    print("\nTesting GARCH data generation...")
    
    generator = GARCHDataGenerator(seed=42)
    
    # Test univariate
    data, metadata = generator.generate(batch_size=5, seq_len=100, num_variates=1)
    print(f"Univariate GARCH shape: {data.shape}")
    print(f"Metadata: {metadata['type']}")
    
    # Test multivariate
    data_multi, _ = generator.generate(batch_size=5, seq_len=100, num_variates=3, correlated=True)
    print(f"Multivariate GARCH shape: {data_multi.shape}")
    
    return True

def test_synthetic_dataset():
    """Test SyntheticDataset class."""
    print("\nTesting SyntheticDataset...")
    
    # Test Wavelet dataset
    dataset = SyntheticDataset(
        data_type="wavelet",
        num_samples=100,
        seq_len=64,
        num_variates=2,
        correlated=False,
        forecast_length=16,
        missing_prob=0.1
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Time series shape: {sample['time_series'].shape}")
    print(f"Forecast mask shape: {sample['forecast_mask'].shape}")
    print(f"Missing mask shape: {sample['missing_mask'].shape}")
    print(f"Variate IDs shape: {sample['variate_ids'].shape}")
    
    # Test DataLoader
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch = next(iter(dataloader))
    print(f"Batch time series shape: {batch['time_series'].shape}")
    
    return True

def test_model_with_synthetic_data():
    """Test Delphyne model with synthetic data."""
    print("\nTesting Delphyne model with synthetic data...")
    
    # Create small config for testing
    config = DelphyneConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        intermediate_size=256,
        patch_size=8,
        max_sequence_length=128,
        num_mixture_components=2
    )
    
    model = DelphyneModel(config)
    
    # Create synthetic dataset
    dataset = SyntheticDataset(
        data_type="wavelet",
        num_samples=32,
        seq_len=64,
        num_variates=1,
        forecast_length=16
    )
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(dataloader))
    
    # Test forward pass
    try:
        outputs = model(
            time_series=batch['time_series'],
            variate_ids=batch['variate_ids'],
            forecast_mask=batch['forecast_mask'],
            targets=batch['time_series'],  # Use same data as targets for testing
            return_dict=True
        )
        
        print("✓ Model forward pass with synthetic data successful")
        print(f"Loss: {outputs['loss'].item():.4f}")
        
        # Test forecast generation
        forecasts = model.generate_forecasts(
            time_series=batch['time_series'],
            forecast_length=16,
            num_samples=5
        )
        
        print(f"Forecast samples shape: {forecasts['samples'].shape}")
        print("✓ Forecast generation successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def visualize_synthetic_data():
    """Visualize synthetic data examples."""
    print("\nGenerating visualization of synthetic data...")
    
    # Generate Wavelet data
    wavelet_gen = WaveletDataGenerator(seed=42)
    wavelet_data, _ = wavelet_gen.generate(batch_size=3, seq_len=200, num_variates=1)
    
    # Generate GARCH data
    garch_gen = GARCHDataGenerator(seed=42)
    garch_data, _ = garch_gen.generate(batch_size=3, seq_len=200, num_variates=1)
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Plot Wavelet data
    for i in range(3):
        axes[0, i].plot(wavelet_data[i].numpy())
        axes[0, i].set_title(f'Wavelet Series {i+1}')
        axes[0, i].set_xlabel('Time')
        axes[0, i].set_ylabel('Value')
    
    # Plot GARCH data
    for i in range(3):
        axes[1, i].plot(garch_data[i].numpy())
        axes[1, i].set_title(f'GARCH Series {i+1}')
        axes[1, i].set_xlabel('Time')
        axes[1, i].set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig('synthetic_data_examples.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved as 'synthetic_data_examples.png'")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("SYNTHETIC DATA TESTING")
    print("=" * 60)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    success = True
    
    # Run tests
    success &= test_wavelet_generation()
    success &= test_garch_generation()
    success &= test_synthetic_dataset()
    success &= test_model_with_synthetic_data()
    
    try:
        success &= visualize_synthetic_data()
    except ImportError:
        print("Matplotlib not available, skipping visualization")
    
    if success:
        print("\n*** ALL SYNTHETIC DATA TESTS PASSED! ***")
    else:
        print("\n*** SOME TESTS FAILED ***")
