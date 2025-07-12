# Bloomberg Forex Trading Strategy - FNAC Implementation

Enhanced implementation of the Fitted Natural Actor-Critic (FNAC) algorithm for Forex trading, based on paper 2410.23294v1, with Bloomberg Terminal integration.

## Features

### Core Algorithm
- **Fitted Natural Actor-Critic (FNAC)**: Advanced reinforcement learning for Forex trading
- **Multi-layer Architecture**: XGBoost value function + Ridge advantage function + Neural network policy
- **Action Spaces**: Both discrete (-1, 0, 1) and continuous [-1, 1] portfolio allocations
- **Persistence Modeling**: Configurable action persistence (1, 5, 10 minutes)

### Bloomberg Integration
- **Real Market Data**: Direct integration with Bloomberg Terminal via xbbg
- **Multiple Currency Pairs**: EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD
- **Intraday Data**: 1-minute resolution with bid/ask spreads
- **Volume Analysis**: Volume-weighted features and patterns
- **Market Microstructure**: Realistic spread modeling based on time-of-day

### Risk Management
- **Risk-Averse Variants**: 
  - Mean-Volatility optimization
  - Risk-Conditional Value at Risk (RCVaR)
- **Transaction Costs**: Fixed and variable fee structures
- **Position Management**: Automatic position closing at day end

### Enhanced Features
- **Dynamic State Space**: Automatically adjusts to available Bloomberg features
- **Multi-Currency Testing**: Batch testing across currency pairs
- **Model Persistence**: Save/load trained models
- **Comprehensive Reporting**: Detailed analysis and performance reports
- **Fallback Mode**: Synthetic data generation when Bloomberg unavailable

## Installation

### Prerequisites
- Bloomberg Terminal access (for real data)
- Python 3.8+

### Setup
```bash
# Clone or download the files
# Install basic dependencies
pip install -r requirements.txt

# For Bloomberg access (optional - synthetic data works without this):
pip install ruamel.yaml
pip install xbbg
pip install blpapi --index-url=https://bcms.bloomberg.com/pip/simple/

# Note: Bloomberg Terminal must be running for real data access
```

## Usage

### Quick Demo (Recommended First Step)
```bash
python demo_strategy.py
```

### Single Currency Analysis (Default: EUR/USD)
```bash
python 2410.23294v1_test_strategy.py
```

### Multi-Currency Testing
```bash
python 2410.23294v1_test_strategy.py --multi-currency
```

### Test Synthetic Data Mode
```bash
python test_synthetic_data.py
```

### Configuration
Edit the main() function to customize:
- `USE_BLOOMBERG`: Set to False for synthetic data
- `CURRENCY_PAIR`: Bloomberg ticker (e.g., "EURUSD Curncy")
- `START_DATE` / `END_DATE`: Data period
- Training parameters (episodes, batch_size, etc.)

## Output

### Generated Files
- `fnac_continuous_model_YYYYMMDD_HHMMSS.pkl`: Trained model
- `bloomberg_forex_report_YYYYMMDD_HHMMSS.txt`: Comprehensive analysis report

### Console Output
- Data summary and validation
- Training progress with episode returns
- Model performance comparison
- Bloomberg-specific feature analysis
- Best model identification

## Algorithm Details

### State Space (49+ dimensions)
- 45 lagged price variations
- Bid/ask spread
- Time features (weekday, minute-of-day)
- Current portfolio position
- Bloomberg features (volume, VWAP deviation, high-low range)

### Action Space
- **Discrete**: {-1, 0, 1} (short, flat, long)
- **Continuous**: [-1, 1] (portfolio allocation)

### Reward Function
```
reward = portfolio_return - transaction_costs
```
With optional risk adjustments for Mean-Volatility or RCVaR

### Training Process
1. **Value Function**: XGBoost regression on state-reward pairs
2. **Advantage Function**: Ridge regression with compatible features
3. **Policy Update**: Neural network with natural gradient approximation

## Bloomberg Data Requirements

### Minimum Access
- Bloomberg Terminal with API access
- xbbg library properly configured
- Sufficient data history (recommended: 2+ years)

### Supported Tickers
- Major currency pairs with "Curncy" suffix
- Intraday data availability varies by Bloomberg subscription

## Performance Notes

### Computational Requirements
- Training time: 5-30 minutes depending on data size
- Memory usage: 1-4 GB for typical datasets
- CPU intensive during XGBoost training phases

### Data Limitations
- Bloomberg intraday data has historical limits
- Weekend/holiday gaps handled automatically
- Synthetic fallback ensures continuous operation

## Troubleshooting

### Bloomberg Connection Issues
```python
# Test Bloomberg connection
import xbbg
data = xbbg.bdh('EURUSD Curncy', 'PX_LAST', '2023-01-01', '2023-01-31')
print(data.head())
```

### Memory Issues
- Reduce date range
- Decrease episode count
- Use smaller batch sizes

### Performance Issues
- Enable synthetic data mode
- Reduce feature dimensions
- Optimize XGBoost parameters

## License

Based on academic research paper 2410.23294v1. Enhanced implementation for educational and research purposes.

## Citation

If using this implementation in research, please cite the original paper:
```
@article{forex_fnac_2024,
  title={Fitted Natural Actor-Critic for Forex Trading},
  journal={arXiv preprint},
  number={2410.23294v1},
  year={2024}
}
```
