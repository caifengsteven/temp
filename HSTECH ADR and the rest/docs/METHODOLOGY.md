# HSTECH Index Estimation Methodology

## Overview

The HSTECH Index Estimation System uses a three-step approach to estimate the HSTECH index price when Hong Kong markets are closed, leveraging US market data and statistical models.

## Three-Step Approach

### Step 1: Static ADR-based HSTECH Update

**Objective**: Calculate direct index updates using dual-listed stocks (ADRs).

**Process**:
1. **Identify ADR Components**: Map HSTECH components to their US ADR equivalents
2. **Price Conversion**: Convert ADR prices to Hong Kong equivalent using:
   - Current USD/HKD exchange rate
   - ADR conversion ratios (shares per ADR)
   - Time zone adjustments
3. **Index Impact Calculation**: Apply component weights to calculate overall index impact

**Key Formulas**:
```
HK_Price = (ADR_Price_USD × Exchange_Rate_USD_HKD) / Conversion_Ratio
Index_Impact = Σ(Price_Change_i × Weight_i) for all ADR components
```

**Coverage**: Approximately 40-50% of HSTECH index by weight (major components like Tencent, Alibaba, JD.com, etc.)

### Step 2: Covariance-based Estimation for Non-dual-listed Stocks

**Objective**: Estimate price movements of non-ADR components using statistical relationships.

**Process**:
1. **Historical Analysis**: Build covariance matrix using 252 days of historical returns
2. **Factor Models**: Use regression models to predict non-ADR movements from ADR movements
3. **Multiple Methods**:
   - **Regression-based**: Ridge regression with regularization
   - **Correlation-weighted**: Weighted average based on historical correlations
   - **Factor model**: Market factor approach using weighted ADR movements

**Key Techniques**:
- Ridge regression to handle multicollinearity
- Minimum correlation threshold (default: 0.3) for reliability
- Cross-validation for model selection
- Ensemble approach combining multiple prediction methods

**Mathematical Foundation**:
```
Non_ADR_Return_i = α_i + Σ(β_ij × ADR_Return_j) + ε_i
```

### Step 3: Enhanced Estimation Using Market Indicators

**Objective**: Incorporate broader market signals to improve estimation accuracy.

**Market Indicators**:
- **PDD Inc. (PDD)**: Chinese e-commerce proxy with high correlation to Chinese tech
- **KWEB ETF**: KraneShares CSI China Internet ETF (broad Chinese internet exposure)
- **ASHR ETF**: Xtrackers Harvest CSI 300 China A-Shares ETF
- **FXI ETF**: iShares China Large-Cap ETF
- **MCHI ETF**: iShares MSCI China ETF

**Enhancement Process**:
1. **Signal Extraction**: Calculate price movements of market indicators
2. **Correlation Weighting**: Apply historical correlations with HSTECH
3. **Signal Combination**: Weighted average of indicator signals
4. **Final Adjustment**: Apply enhancement factor to base estimation

**Default Weights**:
- PDD: 25% (direct Chinese e-commerce exposure)
- KWEB: 35% (broad Chinese internet ETF)
- ASHR: 15% (China A-shares)
- FXI: 15% (China large-cap)
- MCHI: 10% (MSCI China)

## Combination Methodology

### Final Estimation Formula

```
Final_Estimate = Base_ADR_Estimate × (1 + Covariance_Adjustment) × (1 + Indicator_Enhancement)
```

Where:
- **Base_ADR_Estimate**: Step 1 result
- **Covariance_Adjustment**: Step 2 weighted impact
- **Indicator_Enhancement**: Step 3 market signal adjustment

### Method Weights (Configurable)

Default configuration:
- ADR-based: 60%
- Covariance-based: 25%
- Market indicators: 15%

### Confidence Scoring

Confidence is calculated based on:
1. **Data Coverage**: Percentage of index covered by available data
2. **Signal Strength**: Magnitude and consistency of market signals
3. **Model Quality**: R² scores of statistical models
4. **Data Freshness**: Age of input data

**Confidence Formula**:
```
Confidence = (Coverage_Score × 0.4) + (Signal_Strength × 0.3) + (Model_Quality × 0.2) + (Freshness_Score × 0.1)
```

## Data Requirements

### Real-time Data
- US ADR prices (Yahoo Finance, Alpha Vantage)
- Market indicator prices (ETFs, individual stocks)
- USD/HKD exchange rate
- Last known Hong Kong closing prices

### Historical Data
- 252 days of daily returns for all HSTECH components
- Correlation matrices and covariance relationships
- Model training and validation data

## Quality Controls

### Data Validation
- Price reasonableness checks
- Data freshness validation (< 24 hours)
- Missing data handling and fallback mechanisms
- Cross-source validation

### Model Validation
- Backtesting against historical HSTECH performance
- Out-of-sample testing
- Rolling window validation
- Stress testing under market volatility

### Error Handling
- Graceful degradation when data is missing
- Fallback to simpler models if complex models fail
- Confidence adjustment based on data quality
- Logging and monitoring of estimation quality

## Limitations and Assumptions

### Key Assumptions
1. **Market Efficiency**: US ADR prices reflect fair value of underlying HK stocks
2. **Stable Relationships**: Historical correlations remain relevant for short-term prediction
3. **Currency Stability**: Exchange rate movements are captured in real-time
4. **Market Indicators**: Selected ETFs and stocks are representative of Chinese tech sector

### Known Limitations
1. **Coverage Gap**: ~50-60% of index not directly covered by ADRs
2. **Model Risk**: Statistical relationships may break down during market stress
3. **Time Lag**: Some data sources may have delays
4. **Market Microstructure**: Ignores bid-ask spreads and liquidity differences

### Risk Mitigation
- Conservative confidence scoring
- Multiple estimation methods for robustness
- Regular model retraining and validation
- Clear documentation of uncertainty ranges

## Performance Metrics

### Accuracy Metrics
- **Mean Absolute Error (MAE)**: Average absolute difference
- **Mean Absolute Percentage Error (MAPE)**: Percentage-based accuracy
- **Root Mean Squared Error (RMSE)**: Penalizes large errors
- **Correlation**: Linear relationship with actual values

### Directional Accuracy
- Percentage of correct direction predictions
- Critical for trading and investment decisions

### Confidence Calibration
- How well confidence scores correlate with actual accuracy
- Ensures reliable uncertainty quantification

## Implementation Notes

### Computational Complexity
- Real-time estimation: O(n) where n = number of components
- Historical model building: O(n²) for covariance matrix
- Memory usage: Moderate (historical data storage)

### Scalability
- Designed for real-time operation
- Configurable update frequencies
- Efficient caching mechanisms
- Asynchronous data fetching

### Extensibility
- Modular design allows easy addition of new indicators
- Configurable weights and parameters
- Plugin architecture for new estimation methods
