"""
Synthetic data generation for testing Delphyne model.
Implements Wavelet and GARCH data as described in the paper.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from torch.utils.data import Dataset


class WaveletDataGenerator:
    """
    Generate synthetic Wavelet data as described in the paper.
    
    From the paper: x_t = (d*t/T - c) * sin(a*t + b) + ε_t
    where ε_t ~ N(0, σ²)
    """
    
    def __init__(
        self,
        a_range: Tuple[float, float] = (0.1, 2.0),
        b_range: Tuple[float, float] = (0.0, 2*np.pi),
        c_range: Tuple[float, float] = (0.5, 2.0),
        d_range: Tuple[float, float] = (0.5, 2.0),
        noise_std: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Args:
            a_range: Range for frequency parameter a
            b_range: Range for phase parameter b  
            c_range: Range for offset parameter c
            d_range: Range for amplitude parameter d
            noise_std: Standard deviation of Gaussian noise
            seed: Random seed for reproducibility
        """
        self.a_range = a_range
        self.b_range = b_range
        self.c_range = c_range
        self.d_range = d_range
        self.noise_std = noise_std
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    def generate(
        self, 
        batch_size: int, 
        seq_len: int, 
        num_variates: int = 1,
        correlated: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate Wavelet time series data.
        
        Args:
            batch_size: Number of time series to generate
            seq_len: Length of each time series
            num_variates: Number of variates per time series
            correlated: Whether variates should be correlated
            
        Returns:
            Tuple of (data, metadata)
            - data: [batch_size, num_variates, seq_len] if multivariate, [batch_size, seq_len] if univariate
            - metadata: Dictionary with generation parameters
        """
        t = torch.arange(seq_len, dtype=torch.float32)
        T = seq_len
        
        if num_variates == 1:
            # Univariate case
            data = torch.zeros(batch_size, seq_len)
            params = []
            
            for i in range(batch_size):
                # Sample parameters
                a = np.random.uniform(*self.a_range)
                b = np.random.uniform(*self.b_range)
                c = np.random.uniform(*self.c_range)
                d = np.random.uniform(*self.d_range)
                
                # Generate wavelet
                wavelet = (d * t / T - c) * torch.sin(a * t + b)
                
                # Add noise
                noise = torch.randn(seq_len) * self.noise_std
                data[i] = wavelet + noise
                
                params.append({'a': a, 'b': b, 'c': c, 'd': d})
        
        else:
            # Multivariate case
            data = torch.zeros(batch_size, num_variates, seq_len)
            params = []
            
            for i in range(batch_size):
                batch_params = []
                
                if correlated:
                    # Generate correlated variates (same base wavelet + different noise)
                    a = np.random.uniform(*self.a_range)
                    b = np.random.uniform(*self.b_range)
                    c = np.random.uniform(*self.c_range)
                    d = np.random.uniform(*self.d_range)
                    
                    base_wavelet = (d * t / T - c) * torch.sin(a * t + b)
                    
                    for v in range(num_variates):
                        noise = torch.randn(seq_len) * self.noise_std
                        data[i, v] = base_wavelet + noise
                        batch_params.append({'a': a, 'b': b, 'c': c, 'd': d})
                
                else:
                    # Generate uncorrelated variates (different wavelets)
                    for v in range(num_variates):
                        a = np.random.uniform(*self.a_range)
                        b = np.random.uniform(*self.b_range)
                        c = np.random.uniform(*self.c_range)
                        d = np.random.uniform(*self.d_range)
                        
                        wavelet = (d * t / T - c) * torch.sin(a * t + b)
                        noise = torch.randn(seq_len) * self.noise_std
                        data[i, v] = wavelet + noise
                        
                        batch_params.append({'a': a, 'b': b, 'c': c, 'd': d})
                
                params.append(batch_params)
        
        metadata = {
            'type': 'wavelet',
            'params': params,
            'correlated': correlated,
            'noise_std': self.noise_std,
            'seq_len': seq_len,
            'num_variates': num_variates
        }
        
        return data, metadata


class GARCHDataGenerator:
    """
    Generate synthetic GARCH data as described in the paper.
    
    GARCH(1,1): 
    r_t = σ_t * ε_t
    σ_t² = ω + α * r_{t-1}² + β * σ_{t-1}²
    
    where ε_t ~ N(0, 1)
    """
    
    def __init__(
        self,
        omega: float = 0.01,
        alpha: float = 0.1,
        beta: float = 0.8,
        seed: Optional[int] = None
    ):
        """
        Args:
            omega: Constant term in GARCH equation
            alpha: ARCH coefficient
            beta: GARCH coefficient
            seed: Random seed for reproducibility
        """
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        
        # Ensure stationarity condition
        if alpha + beta >= 1.0:
            raise ValueError("GARCH parameters must satisfy alpha + beta < 1 for stationarity")
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    def generate(
        self, 
        batch_size: int, 
        seq_len: int, 
        num_variates: int = 1,
        correlated: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate GARCH time series data.
        
        Args:
            batch_size: Number of time series to generate
            seq_len: Length of each time series
            num_variates: Number of variates per time series
            correlated: Whether variates should be correlated
            
        Returns:
            Tuple of (data, metadata)
        """
        if num_variates == 1:
            # Univariate case
            data = torch.zeros(batch_size, seq_len)
            
            for i in range(batch_size):
                returns, volatilities = self._generate_garch_series(seq_len)
                data[i] = returns
        
        else:
            # Multivariate case
            data = torch.zeros(batch_size, num_variates, seq_len)
            
            for i in range(batch_size):
                if correlated:
                    # Generate one base series and add different noise to create correlation
                    base_returns, _ = self._generate_garch_series(seq_len)
                    
                    for v in range(num_variates):
                        # Add some correlation by mixing with base series
                        correlation = 0.7
                        independent_returns, _ = self._generate_garch_series(seq_len)
                        data[i, v] = correlation * base_returns + (1 - correlation) * independent_returns
                
                else:
                    # Generate independent series
                    for v in range(num_variates):
                        returns, _ = self._generate_garch_series(seq_len)
                        data[i, v] = returns
        
        metadata = {
            'type': 'garch',
            'omega': self.omega,
            'alpha': self.alpha,
            'beta': self.beta,
            'correlated': correlated,
            'seq_len': seq_len,
            'num_variates': num_variates
        }
        
        return data, metadata
    
    def _generate_garch_series(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a single GARCH time series."""
        returns = torch.zeros(seq_len)
        volatilities = torch.zeros(seq_len)
        
        # Initialize
        volatilities[0] = torch.sqrt(torch.tensor(self.omega / (1 - self.alpha - self.beta)))
        returns[0] = volatilities[0] * torch.randn(1)
        
        # Generate series
        for t in range(1, seq_len):
            # Update volatility
            volatilities[t] = torch.sqrt(
                self.omega + self.alpha * returns[t-1]**2 + self.beta * volatilities[t-1]**2
            )
            
            # Generate return
            returns[t] = volatilities[t] * torch.randn(1)
        
        return returns, volatilities


class SyntheticDataset(Dataset):
    """
    PyTorch Dataset for synthetic time series data.
    """
    
    def __init__(
        self,
        data_type: str = "wavelet",
        num_samples: int = 1000,
        seq_len: int = 512,
        num_variates: int = 1,
        correlated: bool = False,
        forecast_length: int = 32,
        missing_prob: float = 0.0,
        **generator_kwargs
    ):
        """
        Args:
            data_type: Type of data to generate ("wavelet" or "garch")
            num_samples: Number of samples in dataset
            seq_len: Length of each time series
            num_variates: Number of variates per time series
            correlated: Whether variates should be correlated
            forecast_length: Length of forecast horizon
            missing_prob: Probability of missing values
            **generator_kwargs: Additional arguments for data generators
        """
        self.data_type = data_type
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_variates = num_variates
        self.forecast_length = forecast_length
        self.missing_prob = missing_prob
        
        # Create data generator
        if data_type == "wavelet":
            self.generator = WaveletDataGenerator(**generator_kwargs)
        elif data_type == "garch":
            self.generator = GARCHDataGenerator(**generator_kwargs)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        # Pre-generate all data
        self.data, self.metadata = self.generator.generate(
            batch_size=num_samples,
            seq_len=seq_len,
            num_variates=num_variates,
            correlated=correlated
        )
        
        # Create forecast masks
        self.forecast_masks = self._create_forecast_masks()
        
        # Create missing masks if needed
        if missing_prob > 0:
            self.missing_masks = self._create_missing_masks()
        else:
            self.missing_masks = None
    
    def _create_forecast_masks(self) -> torch.Tensor:
        """Create forecast masks for each sample."""
        masks = torch.zeros(self.num_samples, self.seq_len)
        masks[:, -self.forecast_length:] = 1.0
        return masks
    
    def _create_missing_masks(self) -> torch.Tensor:
        """Create random missing data masks."""
        masks = torch.rand(self.num_samples, self.seq_len) < self.missing_prob
        return masks.float()
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = {
            'time_series': self.data[idx],
            'forecast_mask': self.forecast_masks[idx]
        }
        
        if self.missing_masks is not None:
            sample['missing_mask'] = self.missing_masks[idx]
        
        # Create variate IDs
        if self.num_variates > 1:
            variate_ids = torch.arange(self.num_variates).unsqueeze(1).repeat(1, self.seq_len)
            sample['variate_ids'] = variate_ids.flatten()
        else:
            sample['variate_ids'] = torch.zeros(self.seq_len, dtype=torch.long)
        
        return sample
