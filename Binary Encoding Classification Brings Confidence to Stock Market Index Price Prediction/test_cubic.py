#!/usr/bin/env python3
"""
Simple test script for CUBIC framework
Tests basic functionality without complex data processing
"""

import torch
import numpy as np
import logging
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cubic.models.cubic_model import CUBICModel
from cubic.utils.binary_encoder import BinaryEncoder
from cubic.utils.confidence_measures import ConfidenceMeasures

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_binary_encoder():
    """Test binary encoder functionality"""
    logger.info("Testing Binary Encoder...")
    
    encoder = BinaryEncoder(precision_bits=15, value_range=(-1, 1))
    
    # Test values
    test_values = [0.0, 0.5, -0.3, 0.8, -0.9, 0.123, -0.456]
    
    print("\nBinary Encoding Test:")
    print(f"{'Original':>10s} {'Encoded (first 8 bits)':>25s} {'Decoded':>10s} {'Error':>10s}")
    print("-" * 60)
    
    for val in test_values:
        encoded = encoder.encode_value(val)
        decoded = encoder.decode_binary(encoded)
        error = abs(val - decoded)
        encoded_str = ''.join(map(str, encoded[:8])) + '...'
        print(f"{val:10.4f} {encoded_str:>25s} {decoded:10.4f} {error:10.6f}")
    
    # Test batch encoding
    batch_values = np.array(test_values)
    batch_encoded = encoder.encode_batch(batch_values)
    batch_decoded = encoder.decode_batch(batch_encoded)
    
    print(f"\nBatch encoding test:")
    print(f"Original: {batch_values}")
    print(f"Decoded:  {batch_decoded}")
    print(f"Max error: {np.max(np.abs(batch_values - batch_decoded)):.6f}")
    
    logger.info("Binary Encoder test completed ✓")


def test_cubic_model():
    """Test CUBIC model functionality"""
    logger.info("Testing CUBIC Model...")
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Model parameters
    batch_size = 8
    sequence_length = 5
    n_stocks = 10
    n_features = 16
    
    # Create synthetic data
    features = torch.randn(batch_size, sequence_length, n_stocks, n_features)
    targets = torch.randn(batch_size) * 0.02  # Small returns
    
    print(f"\nInput shapes:")
    print(f"Features: {features.shape}")
    print(f"Targets: {targets.shape}")
    
    # Create CUBIC model
    model = CUBICModel(
        input_dim=n_features,
        n_stocks=n_stocks,
        backbone_type='lstm',
        backbone_config={'hidden_size': 32, 'num_layers': 1, 'dropout': 0.1},
        fusion_config={'embedding_dim': 16, 'pooling_type': 'multi_head'},
        binary_config={'precision_bits': 15, 'value_range': (-1, 1)}
    )
    
    print(f"\nModel info: {model.get_model_info()}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        # Test basic forward pass
        probabilities = model.forward(features)
        print(f"Output probabilities shape: {probabilities.shape}")
        
        # Test forward pass with confidence
        probabilities, confidence_dict = model.forward(features, return_confidence=True)
        print(f"Mean confidence: {torch.mean(confidence_dict['mean_confidence']):.4f}")
        print(f"Trend confidence: {torch.mean(confidence_dict['trend_confidence']):.4f}")
        
        # Test value prediction
        predicted_values = model.predict_values(features)
        print(f"Predicted values shape: {predicted_values.shape}")
        print(f"Predicted values range: [{torch.min(predicted_values):.4f}, {torch.max(predicted_values):.4f}]")
        
        # Test loss calculation
        total_loss, loss_components = model.calculate_loss(features, targets)
        print(f"\nLoss components:")
        for key, value in loss_components.items():
            print(f"  {key}: {value:.4f}")
    
    logger.info("CUBIC Model test completed ✓")


def test_confidence_measures():
    """Test confidence measures"""
    logger.info("Testing Confidence Measures...")
    
    # Create synthetic probabilities
    batch_size = 5
    precision_bits = 15
    
    # Create probabilities that favor class 1 with varying confidence
    probabilities = torch.zeros(batch_size, precision_bits, 2)
    
    # High confidence samples
    probabilities[0, :, 1] = 0.9
    probabilities[0, :, 0] = 0.1
    
    # Medium confidence samples
    probabilities[1, :, 1] = 0.7
    probabilities[1, :, 0] = 0.3
    
    # Low confidence samples
    probabilities[2, :, 1] = 0.55
    probabilities[2, :, 0] = 0.45
    
    # Mixed confidence
    probabilities[3, :5, 1] = 0.9  # High confidence for first 5 bits
    probabilities[3, :5, 0] = 0.1
    probabilities[3, 5:, 1] = 0.6  # Lower confidence for remaining bits
    probabilities[3, 5:, 0] = 0.4
    
    # Random confidence
    probabilities[4] = torch.softmax(torch.randn(precision_bits, 2), dim=-1)
    
    confidence_measures = ConfidenceMeasures(precision_bits)
    
    # Calculate different confidence measures
    mean_confidence = confidence_measures.calculate_geometric_mean_confidence(probabilities)
    trend_confidence = confidence_measures.calculate_trend_confidence(probabilities)
    entropy_confidence = confidence_measures.calculate_entropy_based_confidence(probabilities)
    
    print(f"\nConfidence Measures:")
    print(f"{'Sample':>8s} {'Mean':>8s} {'Trend':>8s} {'Entropy':>8s}")
    print("-" * 35)
    
    for i in range(batch_size):
        print(f"{i+1:8d} {mean_confidence[i]:8.4f} {trend_confidence[i]:8.4f} {entropy_confidence[i]:8.4f}")
    
    logger.info("Confidence Measures test completed ✓")


def test_integration():
    """Test integration of all components"""
    logger.info("Testing Integration...")
    
    # Set random seed
    torch.manual_seed(42)
    
    # Create small dataset
    batch_size = 4
    sequence_length = 3
    n_stocks = 5
    n_features = 8
    
    features = torch.randn(batch_size, sequence_length, n_stocks, n_features)
    targets = torch.randn(batch_size) * 0.01
    
    # Create model
    model = CUBICModel(
        input_dim=n_features,
        n_stocks=n_stocks,
        backbone_type='mlp',
        backbone_config={'hidden_layers': [32, 16], 'dropout': 0.1},
        fusion_config={'embedding_dim': 8, 'pooling_type': 'multi_head'},
        binary_config={'precision_bits': 10, 'value_range': (-0.1, 0.1)}  # Smaller range for testing
    )
    
    # Test training step
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Forward pass
    total_loss, loss_components = model.calculate_loss(features, targets)
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print(f"\nIntegration Test:")
    print(f"Training loss: {total_loss.item():.4f}")
    print(f"Gradient norm: {torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')):.4f}")
    
    # Test evaluation
    model.eval()
    with torch.no_grad():
        probabilities, confidence_dict = model.forward(features, return_confidence=True)
        predictions = model.predict_values(features)
        
        print(f"Predictions vs Targets:")
        for i in range(batch_size):
            print(f"  Sample {i+1}: Pred={predictions[i]:.4f}, Target={targets[i]:.4f}, "
                  f"Confidence={confidence_dict['mean_confidence'][i]:.4f}")
    
    logger.info("Integration test completed ✓")


def main():
    """Run all tests"""
    print("CUBIC Framework - Component Tests")
    print("=" * 50)
    
    try:
        test_binary_encoder()
        print()
        
        test_confidence_measures()
        print()
        
        test_cubic_model()
        print()
        
        test_integration()
        print()
        
        print("=" * 50)
        print("All tests completed successfully! ✓")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
