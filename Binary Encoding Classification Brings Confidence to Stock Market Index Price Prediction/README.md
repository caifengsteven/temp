# CUBIC Framework: Binary Encoding Classification for Stock Market Index Prediction

This repository implements the CUBIC (Component fUsion and Binary encoding classIfication with Confidence) framework as described in the paper:

**"Why Regression? Binary Encoding Classification Brings Confidence to Stock Market Index Price Prediction"**

## Overview

CUBIC is a novel end-to-end framework that transforms stock market index prediction from traditional regression to binary encoding classification. The framework addresses key challenges in financial forecasting through three main innovations:

1. **Fusion in Latent Space**: Adaptive aggregation of constituent stock information using multi-head pooling
2. **Binary Encoding Classification**: Converting continuous price prediction to multiple binary classification tasks
3. **Confidence-Guided Prediction and Trading**: Leveraging classification probabilities for uncertainty quantification and trading decisions

## Key Features

- **Bloomberg Data Integration**: Direct integration with Bloomberg Terminal via xbbg package
- **Technical Indicators**: Comprehensive calculation of 16 technical indicators (trend, oscillator, volatility)
- **Multiple Architectures**: Support for LSTM, Transformer, and MLP backbones
- **Binary Encoding**: 15-bit precision binary representation of continuous values
- **Confidence Measures**: Geometric mean and trend confidence calculations
- **Trading Strategies**: Confidence-guided position sizing and risk management
- **Comprehensive Evaluation**: IC, ICLR, DA, SR, AR metrics as used in the paper

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cubic-framework
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

## Quick Start

### Using Bloomberg Data

```python
from cubic.data.bloomberg_fetcher import BloombergDataFetcher
from cubic.models.cubic_model import CUBICModel
from cubic.training.trainer import CUBICTrainer

# Fetch data from Bloomberg
data_fetcher = BloombergDataFetcher()
data = data_fetcher.get_index_data('SPX', '2020-01-01', '2024-01-01')

# Create CUBIC model
model = CUBICModel(
    input_dim=16,
    n_stocks=30,
    backbone_type='lstm',
    backbone_config={'hidden_size': 128, 'num_layers': 2}
)

# Train model
trainer = CUBICTrainer(model)
trainer.train(train_loader, val_loader)
```

### Using the Example Script

Run the complete example with synthetic data:
```bash
python example_usage.py
```

Run the main experiment script:
```bash
python main.py --index SPX --backbone lstm --mode full
```

## Configuration

The framework uses YAML configuration files. Key settings include:

```yaml
# Data Configuration
data:
  bloomberg:
    timeout: 30000
    max_retries: 3
  indices:
    SPX:
      bloomberg_ticker: "SPX Index"
      constituent_count: 500

# Model Configuration
model:
  binary_encoding:
    precision_bits: 15
    value_range: [-1, 1]
  embedding:
    stock_embedding_dim: 32

# Training Configuration
training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100
```

## Architecture

### CUBIC Model Components

1. **Stock Embedding**: Projects individual stock indicators to latent space
2. **Multi-Head Pooling**: Aggregates stock embeddings using max, mean, and min pooling
3. **Backbone Models**: LSTM, Transformer, or MLP for temporal modeling
4. **Binary Classification Head**: 15 binary classifiers for each bit position
5. **Confidence Measures**: Geometric confidence calculation for uncertainty quantification

### Binary Encoding

The framework converts continuous values to binary representation using:
```
v = -1 + Σ(k=0 to K) γ_k * 2^(-k)
```
where γ_k ∈ {0,1} are binary digits and K=15 for precision.

## Evaluation Metrics

The framework implements all metrics from the paper:

- **IC (Information Coefficient)**: Pearson correlation between predictions and actual returns
- **ICLR (IC to Loss Ratio)**: IC normalized by standard deviation
- **DA (Direction Accuracy)**: Percentage of correct directional predictions
- **SR (Sharpe Ratio)**: Risk-adjusted return measure
- **AR (Annualized Return)**: Investment profitability measure

## Bloomberg Integration

### Prerequisites

1. Bloomberg Terminal with API access
2. xbbg package installed
3. Valid Bloomberg subscription

### Supported Indices

- **SPX**: S&P 500 Index
- **HSI**: Hang Seng Index  
- **SX5E**: Euro Stoxx 50 Index

### Data Fields

The framework fetches the following Bloomberg fields:
- PX_OPEN, PX_HIGH, PX_LOW, PX_LAST (Price data)
- PX_VOLUME (Volume data)
- INDX_MEMBERS (Index constituents)

## Technical Indicators

The framework calculates 16 technical indicators as specified in the paper:

### Trend Indicators
- Arithmetic Ratio, Open, Close
- Close SMA, Volume SMA
- Close EMA, Volume EMA, ADX

### Oscillator Indicators
- RSI, MACD, MACD Signal, K, MFI

### Volatility Indicators
- ATR, BB Middle, OBV

## Confidence-Guided Trading

The framework implements sophisticated trading strategies based on prediction confidence:

### Confidence Measures
- **Mean Confidence**: Geometric mean across all binary digits
- **Trend Confidence**: Confidence in the most significant bit (price direction)

### Position Sizing
- High confidence (>0.9): 100% position
- Medium confidence (0.7-0.9): 75% position  
- Low confidence (0.5-0.7): 50% position
- Very low confidence (<0.5): No position

## Project Structure

```
cubic/
├── data/                   # Data fetching and processing
│   ├── bloomberg_fetcher.py
│   ├── technical_indicators.py
│   └── data_processor.py
├── models/                 # Model architectures
│   ├── fusion_module.py
│   ├── backbones.py
│   └── cubic_model.py
├── utils/                  # Utilities
│   ├── binary_encoder.py
│   ├── confidence_measures.py
│   └── config_manager.py
├── training/               # Training pipeline
│   └── trainer.py
└── evaluation/             # Evaluation metrics
    ├── metrics.py
    └── evaluator.py
```

## Results

The CUBIC framework demonstrates consistent improvements across different markets and model architectures:

- **IC improvements**: Up to 0.076 (vs 0.012 baseline)
- **Sharpe Ratio**: Up to 1.655 (vs 0.584 baseline)
- **Direction Accuracy**: Up to 59.3% (vs 46.4% baseline)
- **Annualized Return**: Up to 17.7% (vs 3.7% baseline)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{cubic2024,
  title={Why Regression? Binary Encoding Classification Brings Confidence to Stock Market Index Price Prediction},
  author={Jiang, Junzhe and Yang, Chang and Wang, Xinrun and Li, Bo},
  journal={arXiv preprint arXiv:2506.03153},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For questions and support, please open an issue on GitHub.
