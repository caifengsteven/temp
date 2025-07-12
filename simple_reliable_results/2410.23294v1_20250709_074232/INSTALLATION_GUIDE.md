# Installation Guide - Bloomberg Forex Trading Strategy

## Quick Start (Synthetic Data Mode)

The strategy works perfectly with synthetic data and requires no Bloomberg setup:

```bash
# 1. Install basic dependencies
pip install numpy pandas scikit-learn xgboost matplotlib

# 2. Run the demo
python demo_strategy.py

# 3. Run the full strategy
python 2410.23294v1_test_strategy.py
```

## Bloomberg Integration Setup

For real market data, follow these steps:

### Step 1: Install Python Dependencies

```bash
# Install required Python packages
pip install ruamel.yaml
pip install xbbg
```

### Step 2: Install Bloomberg API

```bash
# Install Bloomberg Python API (requires Bloomberg Terminal access)
pip install blpapi --index-url=https://bcms.bloomberg.com/pip/simple/
```

### Step 3: Bloomberg Terminal Setup

1. **Bloomberg Terminal Access**: You need a valid Bloomberg Terminal subscription
2. **Terminal Running**: Bloomberg Terminal must be running on your machine
3. **API Access**: Ensure your Bloomberg subscription includes API access

### Step 4: Test Bloomberg Connection

```bash
# Test if Bloomberg integration works
python test_bloomberg_connection.py
```

Expected output if working:
```
✓ xbbg library imported successfully
✓ Successfully retrieved X daily data points
✓ Successfully retrieved X intraday data points
```

## Troubleshooting

### Common Issues

#### 1. "No module named 'ruamel'"
```bash
pip install ruamel.yaml
```

#### 2. "No module named 'xbbg'"
```bash
pip install xbbg
```

#### 3. "No module named 'blpapi'"
```bash
pip install blpapi --index-url=https://bcms.bloomberg.com/pip/simple/
```

#### 4. "Bloomberg connection failed"
- Ensure Bloomberg Terminal is running
- Check your Bloomberg subscription includes API access
- Verify you're logged into Bloomberg Terminal

#### 5. "No data returned from Bloomberg"
- Check if the ticker symbol is correct (e.g., "EURUSD Curncy")
- Verify the date range is valid (not weekends/holidays)
- Some data may require special permissions

### Fallback Mode

If Bloomberg setup fails, the strategy automatically falls back to synthetic data:

```
✗ Bloomberg xbbg library not available
Will use synthetic data mode
Generating synthetic Forex data...
✓ Generated 23,955 synthetic data points
```

This is perfectly fine for testing and development!

## Verification Steps

### 1. Test Synthetic Data (Always Works)
```bash
python test_synthetic_data.py
```

### 2. Test Bloomberg Connection (If Available)
```bash
python test_bloomberg_connection.py
```

### 3. Run Demo (Shows All Features)
```bash
python demo_strategy.py
```

### 4. Run Full Strategy
```bash
# Single currency
python 2410.23294v1_test_strategy.py

# Multiple currencies (if Bloomberg available)
python 2410.23294v1_test_strategy.py --multi-currency
```

## Configuration

Edit `config.py` to customize:

```python
# Force synthetic data mode
DATA_CONFIG = {
    'use_bloomberg': False,  # Set to True for Bloomberg
    'currency_pair': 'EURUSD Curncy',
    'start_date': '2018-01-01',
    'end_date': '2022-12-31',
}

# Adjust training parameters
TRAINING_CONFIG = {
    'episodes': 50,  # Increase for better performance
    'batch_size': 64,
    'evaluate_every': 5,
}
```

## System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- 1GB disk space

### Recommended for Bloomberg
- Python 3.9+
- 8GB RAM
- Bloomberg Terminal subscription
- Windows/Linux/Mac (Bloomberg Terminal supported OS)

### Performance Notes
- Training time: 5-30 minutes depending on data size
- Memory usage: 1-4GB for typical datasets
- CPU intensive during XGBoost training

## Support

### Working Features (No Bloomberg Required)
- ✅ Synthetic data generation
- ✅ FNAC algorithm training
- ✅ Risk management variants
- ✅ Model persistence
- ✅ Performance analysis

### Bloomberg-Dependent Features
- ⚠️ Real market data
- ⚠️ Multi-currency real data
- ⚠️ Real-time features

### Getting Help

1. **Check the demo**: `python demo_strategy.py`
2. **Review logs**: Look for error messages in console output
3. **Test components**: Use individual test scripts
4. **Configuration**: Verify `config.py` settings

## Success Indicators

You'll know everything is working when:

1. **Demo runs successfully**:
   ```
   🎉 Demo completed successfully!
   ```

2. **Strategy trains without errors**:
   ```
   Episode 50/50, Train Return: X.XX, Validation Return: Y.YY
   Final Test Return: Z.ZZ
   ```

3. **Models are saved**:
   ```
   Model saved to fnac_continuous_model_YYYYMMDD_HHMMSS.pkl
   ```

4. **Reports are generated**:
   ```
   Report saved to bloomberg_forex_report_YYYYMMDD_HHMMSS.txt
   ```

The strategy is designed to work perfectly with synthetic data, so Bloomberg integration is optional for learning and testing the algorithm!
