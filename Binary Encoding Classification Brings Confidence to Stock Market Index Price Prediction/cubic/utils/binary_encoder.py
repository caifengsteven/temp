"""
Binary Encoder for CUBIC framework
Implements the binary encoding classification approach from the paper
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)


class BinaryEncoder:
    """
    Binary encoding system for converting continuous values to binary representations
    As described in the CUBIC paper: v = -1 + Σ(k=0 to K) γ_k * 2^(-k)
    """
    
    def __init__(self, precision_bits: int = 15, value_range: Tuple[float, float] = (-1, 1)):
        """
        Initialize Binary Encoder
        
        Args:
            precision_bits: Number of bits for binary representation (K in the paper)
            value_range: Range of values to encode (default: [-1, 1])
        """
        self.precision_bits = precision_bits
        self.value_range = value_range
        self.min_val, self.max_val = value_range
        
        # Calculate precision
        self.precision = 2 ** (-precision_bits)
        
        logger.info(f"Binary encoder initialized with {precision_bits} bits, "
                   f"range {value_range}, precision {self.precision}")
    
    def normalize_value(self, value: float) -> float:
        """
        Normalize value to [-1, 1] range
        
        Args:
            value: Input value
            
        Returns:
            Normalized value in [-1, 1]
        """
        # Clip to range
        value = np.clip(value, self.min_val, self.max_val)
        
        # Normalize to [-1, 1]
        normalized = 2 * (value - self.min_val) / (self.max_val - self.min_val) - 1
        
        return normalized
    
    def denormalize_value(self, normalized_value: float) -> float:
        """
        Denormalize value from [-1, 1] range back to original range
        
        Args:
            normalized_value: Value in [-1, 1] range
            
        Returns:
            Denormalized value
        """
        # Convert from [-1, 1] to [0, 1]
        unit_value = (normalized_value + 1) / 2
        
        # Scale to original range
        original_value = unit_value * (self.max_val - self.min_val) + self.min_val
        
        return original_value
    
    def encode_value(self, value: float) -> List[int]:
        """
        Encode a continuous value to binary representation
        
        Args:
            value: Continuous value to encode
            
        Returns:
            List of binary digits (0 or 1)
        """
        # Normalize to [-1, 1]
        normalized = self.normalize_value(value)
        
        # Ensure value is in valid range for encoding
        normalized = np.clip(normalized, -1, 1 - self.precision)
        
        # Convert to binary using the paper's formula: v = -1 + Σ(k=0 to K) γ_k * 2^(-k)
        # Rearranging: Σ(k=0 to K) γ_k * 2^(-k) = v + 1
        target_sum = normalized + 1
        
        binary_digits = []
        remaining = target_sum
        
        for k in range(self.precision_bits):
            bit_value = 2 ** (-k)
            
            if remaining >= bit_value:
                binary_digits.append(1)
                remaining -= bit_value
            else:
                binary_digits.append(0)
        
        return binary_digits
    
    def decode_binary(self, binary_digits: List[int]) -> float:
        """
        Decode binary representation back to continuous value
        
        Args:
            binary_digits: List of binary digits
            
        Returns:
            Decoded continuous value
        """
        if len(binary_digits) != self.precision_bits:
            raise ValueError(f"Expected {self.precision_bits} binary digits, got {len(binary_digits)}")
        
        # Calculate sum using the paper's formula: v = -1 + Σ(k=0 to K) γ_k * 2^(-k)
        binary_sum = sum(bit * (2 ** (-k)) for k, bit in enumerate(binary_digits))
        normalized_value = -1 + binary_sum
        
        # Denormalize to original range
        original_value = self.denormalize_value(normalized_value)
        
        return original_value
    
    def encode_batch(self, values: np.ndarray) -> np.ndarray:
        """
        Encode a batch of values to binary representations
        
        Args:
            values: Array of continuous values
            
        Returns:
            Array of binary representations with shape (batch_size, precision_bits)
        """
        batch_size = len(values)
        binary_batch = np.zeros((batch_size, self.precision_bits), dtype=np.int32)
        
        for i, value in enumerate(values):
            binary_batch[i] = self.encode_value(value)
        
        return binary_batch
    
    def decode_batch(self, binary_batch: np.ndarray) -> np.ndarray:
        """
        Decode a batch of binary representations to continuous values
        
        Args:
            binary_batch: Array of binary representations
            
        Returns:
            Array of decoded continuous values
        """
        batch_size = binary_batch.shape[0]
        values = np.zeros(batch_size)
        
        for i in range(batch_size):
            values[i] = self.decode_binary(binary_batch[i].tolist())
        
        return values
    
    def get_position_weights(self, weight_type: str = "exponential") -> torch.Tensor:
        """
        Get position-dependent weights for binary digits
        
        Args:
            weight_type: Type of weighting ("exponential", "linear", "uniform")
            
        Returns:
            Tensor of weights for each binary position
        """
        if weight_type == "exponential":
            # Higher weights for more significant bits
            weights = torch.tensor([2 ** (-k) for k in range(self.precision_bits)])
        elif weight_type == "linear":
            # Linear decrease in weights
            weights = torch.tensor([(self.precision_bits - k) / self.precision_bits 
                                  for k in range(self.precision_bits)])
        elif weight_type == "uniform":
            # Equal weights for all positions
            weights = torch.ones(self.precision_bits)
        else:
            raise ValueError(f"Unknown weight type: {weight_type}")
        
        return weights
    
    def create_binary_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous targets to binary classification targets
        
        Args:
            targets: Continuous target values
            
        Returns:
            Binary targets with shape (batch_size, precision_bits)
        """
        batch_size = targets.shape[0]
        binary_targets = torch.zeros(batch_size, self.precision_bits, dtype=torch.long)
        
        for i, target in enumerate(targets):
            binary_digits = self.encode_value(target.item())
            binary_targets[i] = torch.tensor(binary_digits, dtype=torch.long)
        
        return binary_targets
    
    def reconstruct_from_probabilities(self, probabilities: torch.Tensor, 
                                     use_argmax: bool = True) -> torch.Tensor:
        """
        Reconstruct continuous values from binary classification probabilities
        
        Args:
            probabilities: Probabilities for each binary digit (batch_size, precision_bits, 2)
            use_argmax: Whether to use argmax for binary decision or use probabilities directly
            
        Returns:
            Reconstructed continuous values
        """
        batch_size = probabilities.shape[0]
        reconstructed = torch.zeros(batch_size)
        
        if use_argmax:
            # Use argmax to get binary decisions
            binary_predictions = torch.argmax(probabilities, dim=-1)
            
            for i in range(batch_size):
                binary_digits = binary_predictions[i].cpu().numpy().tolist()
                reconstructed[i] = self.decode_binary(binary_digits)
        else:
            # Use probabilities directly (soft reconstruction)
            for i in range(batch_size):
                soft_sum = 0
                for k in range(self.precision_bits):
                    # Use probability of bit being 1
                    bit_prob = probabilities[i, k, 1]
                    soft_sum += bit_prob * (2 ** (-k))
                
                normalized_value = -1 + soft_sum
                reconstructed[i] = self.denormalize_value(normalized_value.item())
        
        return reconstructed
    
    def calculate_reconstruction_error(self, original: torch.Tensor, 
                                     reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Calculate reconstruction error between original and reconstructed values
        
        Args:
            original: Original continuous values
            reconstructed: Reconstructed values from binary encoding
            
        Returns:
            Mean squared error between original and reconstructed values
        """
        mse = torch.mean((original - reconstructed) ** 2)
        return mse
    
    def get_encoding_info(self) -> dict:
        """
        Get information about the binary encoding configuration
        
        Returns:
            Dictionary with encoding information
        """
        return {
            'precision_bits': self.precision_bits,
            'value_range': self.value_range,
            'precision': self.precision,
            'max_representable_values': 2 ** self.precision_bits,
            'theoretical_precision': self.precision
        }
