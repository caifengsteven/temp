"""
Test script for Delphyne model implementation
"""

import torch
import numpy as np
from delphyne import DelphyneModel, DelphyneConfig

def test_basic_functionality():
    """Test basic model functionality with synthetic data."""
    print("Testing Delphyne model basic functionality...")
    
    # Create a small configuration for testing
    config = DelphyneConfig(
        num_layers=2,           # Smaller for testing
        hidden_size=128,        # Smaller for testing
        num_attention_heads=4,  # Smaller for testing
        intermediate_size=512,  # Smaller for testing
        patch_size=8,          # Smaller for testing
        max_sequence_length=256, # Smaller for testing
        context_length=32,      # Smaller for testing
        num_mixture_components=2 # Smaller for testing
    )
    
    print(f"Config: {config}")
    
    # Create model
    model = DelphyneModel(config)
    print(f"Model created with {model.get_num_parameters():,} parameters")
    
    # Create synthetic data
    batch_size = 2
    seq_len = 64
    num_variates = 3
    
    # Test univariate case
    print("\n--- Testing Univariate Case ---")
    univariate_data = torch.randn(batch_size, seq_len)
    print(f"Input shape: {univariate_data.shape}")
    
    try:
        outputs = model(univariate_data, return_dict=True)
        print("✓ Univariate forward pass successful")
        print(f"Distribution type: {type(outputs['distribution'])}")
        print(f"Last hidden state shape: {outputs['last_hidden_state'].shape}")
        
        # Test sampling
        samples = outputs['distribution'].sample((5,))
        print(f"Sample shape: {samples.shape}")
        
    except Exception as e:
        print(f"✗ Univariate forward pass failed: {e}")
        return False
    
    # Test multivariate case
    print("\n--- Testing Multivariate Case ---")
    multivariate_data = torch.randn(batch_size, num_variates, seq_len)
    variate_ids = torch.arange(num_variates).unsqueeze(0).repeat(seq_len, 1).transpose(0, 1).flatten()
    variate_ids = variate_ids.unsqueeze(0).repeat(batch_size, 1)
    
    print(f"Input shape: {multivariate_data.shape}")
    print(f"Variate IDs shape: {variate_ids.shape}")
    
    try:
        outputs = model(
            time_series=multivariate_data,
            variate_ids=variate_ids,
            return_dict=True
        )
        print("✓ Multivariate forward pass successful")
        print(f"Distribution type: {type(outputs['distribution'])}")
        print(f"Last hidden state shape: {outputs['last_hidden_state'].shape}")
        
    except Exception as e:
        print(f"✗ Multivariate forward pass failed: {e}")
        return False
    
    # Test with targets (loss computation)
    print("\n--- Testing Loss Computation ---")
    targets = torch.randn(batch_size, seq_len)
    forecast_mask = torch.zeros(batch_size, seq_len)
    forecast_mask[:, -16:] = 1.0  # Forecast last 16 positions
    
    try:
        outputs = model(
            time_series=univariate_data,
            targets=targets,
            forecast_mask=forecast_mask,
            return_dict=True
        )
        print("✓ Loss computation successful")
        print(f"Loss: {outputs['loss'].item():.4f}")
        
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        return False
    
    # Test forecast generation
    print("\n--- Testing Forecast Generation ---")
    try:
        forecasts = model.generate_forecasts(
            time_series=univariate_data,
            forecast_length=16,
            num_samples=10
        )
        print("✓ Forecast generation successful")
        print(f"Forecast samples shape: {forecasts['samples'].shape}")
        print(f"Mean forecast shape: {forecasts['mean'].shape}")
        print(f"Quantiles shape: {forecasts['quantiles'].shape}")
        
    except Exception as e:
        print(f"✗ Forecast generation failed: {e}")
        return False
    
    print("\n*** All tests passed! ***")
    return True

def test_attention_mechanism():
    """Test the any-variate attention mechanism specifically."""
    print("\n--- Testing Any-Variate Attention ---")
    
    from delphyne.model.attention import AnyVariateAttention
    
    config = DelphyneConfig(
        hidden_size=64,
        num_attention_heads=4,
        dropout_prob=0.1
    )
    
    attention = AnyVariateAttention(config)
    
    batch_size = 2
    seq_len = 16
    hidden_size = 64
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    variate_ids = torch.randint(0, 5, (batch_size, seq_len))
    
    try:
        outputs = attention(hidden_states, variate_ids, output_attentions=True)
        print("✓ Any-variate attention successful")
        print(f"Output shape: {outputs[0].shape}")
        print(f"Attention weights shape: {outputs[1].shape}")
        
    except Exception as e:
        print(f"✗ Any-variate attention failed: {e}")
        return False
    
    return True

def test_patching():
    """Test the patching mechanism."""
    print("\n--- Testing Patching Mechanism ---")
    
    from delphyne.model.patching import TimeSeriesPatcher
    
    patcher = TimeSeriesPatcher(patch_size=8)
    
    # Test univariate
    univariate_data = torch.randn(2, 32)
    patch_data = patcher(univariate_data)
    
    print(f"Univariate patches shape: {patch_data['patches'].shape}")
    print(f"Variate IDs shape: {patch_data['variate_ids'].shape}")
    
    # Test multivariate
    multivariate_data = torch.randn(2, 3, 32)
    patch_data = patcher(multivariate_data)
    
    print(f"Multivariate patches shape: {patch_data['patches'].shape}")
    print(f"Variate IDs shape: {patch_data['variate_ids'].shape}")
    
    print("✓ Patching mechanism works")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("DELPHYNE MODEL TESTING")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    success = True
    
    # Run tests
    success &= test_patching()
    success &= test_attention_mechanism()
    success &= test_basic_functionality()
    
    if success:
        print("\n*** ALL TESTS PASSED! ***")
        print("The Delphyne model implementation is working correctly.")
    else:
        print("\n*** SOME TESTS FAILED ***")
        print("Please check the implementation.")
