"""
Configuration classes for Delphyne model
"""

from dataclasses import dataclass
from typing import Optional, List, Union
import torch


@dataclass
class DelphyneConfig:
    """Configuration class for Delphyne model
    
    Based on the paper specifications:
    - 12 layers, 768-dimensional attention, 12 heads
    - Maximum width of 3072, dropout 0.2
    - Patch size of 32, sequence length of 512×32 steps
    """
    
    # Model architecture
    num_layers: int = 12
    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    dropout_prob: float = 0.2
    
    # Sequence and patching
    patch_size: int = 32
    max_sequence_length: int = 512 * 32  # 16384
    context_length: int = 512
    
    # Attention configuration
    use_any_variate_attention: bool = True
    rotary_embedding: bool = True
    
    # Output distribution
    num_mixture_components: int = 4  # For Student-T mixture
    
    # Masking configuration
    masking_ratio_alpha: float = 5.0  # Beta-binomial parameters
    masking_ratio_beta: float = 10.0
    average_masking_ratio: float = 0.3
    
    # Training configuration
    vocab_size: Optional[int] = None  # Will be set based on data
    max_position_embeddings: int = 16384
    layer_norm_eps: float = 1e-12
    
    # Activation functions
    hidden_act: str = "silu"  # SiLU activation as specified
    use_glu: bool = True  # Gated Linear Unit
    
    # Initialization
    initializer_range: float = 0.02
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        
        self.head_dim = self.hidden_size // self.num_attention_heads


@dataclass 
class TrainingConfig:
    """Training configuration matching the paper specifications"""
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-8
    
    # Training schedule
    num_train_steps: int = 1_000_000
    warmup_steps: int = 10_000
    batch_size: int = 256
    gradient_accumulation_steps: int = 1
    
    # Mixed precision
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = "bf16"
    
    # Checkpointing
    save_steps: int = 10_000
    eval_steps: int = 5_000
    logging_steps: int = 100
    
    # Data
    max_length: int = 512 * 32
    dataloader_num_workers: int = 4
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging
    log_level: str = "INFO"
    use_wandb: bool = False
    project_name: str = "delphyne"
    
    # Evaluation
    eval_batch_size: int = 64
    eval_accumulation_steps: int = 1


@dataclass
class DataConfig:
    """Data configuration for training and evaluation"""
    
    # Data paths
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    test_data_path: Optional[str] = None
    
    # Data processing
    normalize_per_variate: bool = True
    handle_missing_data: bool = True
    right_padding: bool = True
    
    # Synthetic data (for testing)
    use_synthetic_data: bool = False
    synthetic_data_type: str = "wavelet"  # "wavelet" or "garch"
    num_synthetic_samples: int = 10000
    synthetic_sequence_length: int = 1024
    num_variates: int = 4
    
    # Financial data specific
    include_financial_data: bool = True
    financial_data_ratio: float = 0.5  # Ratio of financial to general data
