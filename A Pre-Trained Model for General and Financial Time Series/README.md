# Delphyne: A Pre-Trained Model for General and Financial Time Series

This repository contains a PyTorch implementation of **Delphyne**, a pre-trained transformer model for time series forecasting, based on the paper "DELPHYNE: A PRE-TRAINED MODEL FOR GENERAL AND FINANCIAL TIMESERIES".

## 🎯 Key Features

- **Any-Variate Attention**: Specialized attention mechanism for multivariate time series
- **Mixture of Student-T Output**: Probabilistic forecasting with heavy-tailed distributions
- **Patch-Based Processing**: Efficient handling of long time series sequences
- **Pre-training Ready**: Supports both general and financial time series data
- **Rotary Positional Embeddings**: Advanced positional encoding for transformers
- **Mixed Precision Training**: Efficient training with bf16 support

## 🏗️ Architecture

Delphyne follows the paper specifications:
- **12 layers** with 768-dimensional attention and 12 heads
- **Patch size of 32** for sequence tokenization
- **GLU with SiLU activation** instead of standard FFN
- **Any-variate attention** for multivariate relationships
- **Student-T mixture output** for robust probabilistic forecasting

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd delphyne

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Usage

```python
import torch
from delphyne import DelphyneModel, DelphyneConfig

# Create model configuration
config = DelphyneConfig(
    num_layers=12,
    hidden_size=768,
    num_attention_heads=12,
    patch_size=32,
    max_sequence_length=16384
)

# Initialize model
model = DelphyneModel(config)

# Generate forecasts
time_series = torch.randn(4, 512)  # [batch_size, seq_len]
forecasts = model.generate_forecasts(
    time_series=time_series,
    forecast_length=64,
    num_samples=100
)

print(f"Forecast samples shape: {forecasts['samples'].shape}")
print(f"Mean forecast: {forecasts['mean'].shape}")
print(f"Quantiles: {forecasts['quantiles'].shape}")
```

### Training on Synthetic Data

```python
from delphyne.data import SyntheticDataset
from delphyne.training import DelphyneTrainer
from torch.utils.data import DataLoader

# Create synthetic dataset
dataset = SyntheticDataset(
    data_type="wavelet",  # or "garch"
    num_samples=1000,
    seq_len=512,
    num_variates=1,
    forecast_length=64
)

# Create data loader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Create trainer
trainer = DelphyneTrainer(
    model=model,
    config=training_config,
    train_dataloader=dataloader
)

# Train model
results = trainer.train()
```

## 📊 Synthetic Data Generation

The implementation includes synthetic data generators as described in the paper:

### Wavelet Data
```python
from delphyne.data import WaveletDataGenerator

generator = WaveletDataGenerator()
data, metadata = generator.generate(
    batch_size=100,
    seq_len=512,
    num_variates=3,
    correlated=True
)
```

### GARCH Data
```python
from delphyne.data import GARCHDataGenerator

generator = GARCHDataGenerator(
    omega=0.01,
    alpha=0.1,
    beta=0.8
)
data, metadata = generator.generate(
    batch_size=100,
    seq_len=512,
    num_variates=1
)
```

## 🔧 Training

### Command Line Training

```bash
python train_delphyne.py \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --num_epochs 10 \
    --mixed_precision \
    --device cuda
```

### Custom Training Configuration

```python
from delphyne import TrainingConfig

config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=256,
    num_train_steps=1_000_000,
    warmup_steps=10_000,
    use_mixed_precision=True,
    mixed_precision_dtype="bf16"
)
```

## 📈 Model Components

### Any-Variate Attention
Specialized attention mechanism that handles relationships between different variates in multivariate time series:

```python
from delphyne.model import AnyVariateAttention

attention = AnyVariateAttention(config)
output = attention(
    hidden_states=embeddings,
    variate_ids=variate_ids
)
```

### Student-T Mixture Output
Probabilistic output layer using mixture of Student-T distributions:

```python
from delphyne.model import StudentTMixtureOutput

output_layer = StudentTMixtureOutput(config)
distribution_params = output_layer(hidden_states)
samples = distribution_params['distribution'].sample((100,))
```

### Time Series Patching
Efficient patching mechanism for long sequences:

```python
from delphyne.model import TimeSeriesPatcher

patcher = TimeSeriesPatcher(patch_size=32)
patch_data = patcher(
    time_series=data,
    variate_ids=variate_ids,
    missing_mask=missing_mask
)
```

## 🧪 Testing

Run the test suite to verify the implementation:

```bash
# Test basic model functionality
python test_model.py

# Test synthetic data generation
python test_synthetic_data.py

# Test training infrastructure
python train_delphyne.py --num_epochs 1 --batch_size 4
```

## 📋 Configuration Options

### Model Configuration
- `num_layers`: Number of transformer layers (default: 12)
- `hidden_size`: Hidden dimension size (default: 768)
- `num_attention_heads`: Number of attention heads (default: 12)
- `patch_size`: Size of time series patches (default: 32)
- `num_mixture_components`: Number of Student-T components (default: 4)

### Training Configuration
- `learning_rate`: AdamW learning rate (default: 1e-4)
- `weight_decay`: Weight decay coefficient (default: 0.1)
- `warmup_steps`: Linear warmup steps (default: 10,000)
- `use_mixed_precision`: Enable bf16 training (default: False)

## 🔬 Paper Implementation Details

This implementation follows the paper specifications:

1. **Architecture**: 12-layer transformer with 768-dim attention
2. **Attention**: Any-variate attention with binary bias terms
3. **Activation**: SiLU activation with Gated Linear Units
4. **Output**: Mixture of Student-T distributions
5. **Training**: AdamW optimizer with cosine annealing
6. **Data**: Patch-based processing with instance normalization

## 📚 References

```bibtex
@article{delphyne2025,
  title={DELPHYNE: A PRE-TRAINED MODEL FOR GENERAL AND FINANCIAL TIMESERIES},
  author={Ding, Xueying and Mittal, Aakriti and Gopal, Achintya},
  journal={arXiv preprint arXiv:2506.06288},
  year={2025}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original paper authors: Xueying Ding, Aakriti Mittal, Achintya Gopal
- PyTorch team for the excellent deep learning framework
- Hugging Face for transformer implementation inspiration
