# Learn2Rank Database Integration Guide

## Overview
This project integrates your MySQL database containing stock factor data with Learn2Rank algorithms for stock ranking and selection. The system can extract data from your `yuqerdata.yq_mktstockfactorsonedayget` table and train ranking models to predict stock performance.

## Database Configuration
- **Host**: localhost
- **User**: root  
- **Password**: 352471Cf
- **Database**: yuqerdata
- **Table**: yq_mktstockfactorsonedayget

## Files Created

### 1. `database_connector.py`
- Connects to MySQL database
- Examines table structure and data
- Extracts sample data for analysis
- **Usage**: `python database_connector.py`

### 2. `start_mysql.py`
- Attempts to start MySQL service automatically
- Provides manual instructions if automatic start fails
- **Usage**: `python start_mysql.py`

### 3. `learn2rank_with_database.py`
- Main pipeline for database integration
- Connects to database and extracts factor data
- Demonstrates Learn2Rank with existing CSV data
- **Usage**: `python learn2rank_with_database.py`

### 4. `train_learn2rank_models.py`
- Comprehensive training pipeline for all algorithms
- Extracts data from database and prepares for ranking
- Trains RankNet, ListMLE, and LambdaMART models
- Saves results and predictions
- **Usage**: `python train_learn2rank_models.py`

### 5. `demo_with_sample_data.py`
- Complete demonstration using generated sample data
- Shows all three algorithms working
- Generates performance metrics and comparisons
- **Usage**: `python demo_with_sample_data.py`

## Learn2Rank Algorithms Implemented

### 1. RankNet
- Neural network-based pairwise ranking
- Uses sigmoid activation for pairwise comparisons
- Good for learning relative rankings

### 2. ListMLE (ListNet)
- Listwise ranking approach
- Optimizes entire ranking list simultaneously
- Uses ListMLE loss function with NDCG evaluation

### 3. LambdaMART
- Gradient boosting for ranking (XGBoost implementation)
- Combines multiple weak learners
- Often performs well in practice

## Current Status

### ✅ Completed
1. **Database Connection Setup**: Scripts ready to connect to your MySQL database
2. **Algorithm Implementation**: All three Learn2Rank algorithms implemented and tested
3. **Data Pipeline**: Complete pipeline from database extraction to model training
4. **Demo System**: Working demonstration with sample data showing all algorithms
5. **Performance Evaluation**: Metrics calculation and comparison framework

### ⚠️ Pending (Requires Your Action)
1. **MySQL Service**: Start MySQL service as administrator
   ```cmd
   # Open Command Prompt as Administrator
   net start MySQL57
   ```

2. **Database Access**: Verify database connection and table access
   ```bash
   python database_connector.py
   ```

3. **Real Data Training**: Once database is accessible, run full training
   ```bash
   python train_learn2rank_models.py
   ```

## Expected Database Schema
The system expects the following columns in `yq_mktstockfactorsonedayget`:
- `ts_code`: Stock code identifier
- `trade_date`: Trading date
- `close`: Closing price
- `volume`: Trading volume
- `market_cap`: Market capitalization
- `pe_ratio`, `pb_ratio`, `ps_ratio`, `pcf_ratio`: Valuation ratios
- `roe`, `roa`: Profitability ratios
- `gross_profit_margin`, `net_profit_margin`: Margin ratios
- `current_ratio`, `quick_ratio`: Liquidity ratios
- `debt_to_equity_ratio`: Leverage ratio
- `revenue_growth`, `eps_growth`: Growth metrics
- `book_value_per_share`, `cash_per_share`: Per-share metrics

## Performance Metrics
The system calculates:
- **Top 100 Return**: Average return of top 100 ranked stocks
- **Bottom 100 Return**: Average return of bottom 100 ranked stocks  
- **Long-Short Return**: Difference between top and bottom (key metric)
- **Win Rate**: Percentage of periods with positive long-short returns
- **NDCG@100**: Normalized Discounted Cumulative Gain for ranking quality

## Next Steps

### Immediate Actions
1. **Start MySQL Service**:
   - Open Command Prompt as Administrator
   - Run: `net start MySQL57`
   - Verify with: `python start_mysql.py`

2. **Test Database Connection**:
   ```bash
   python database_connector.py
   ```

3. **Examine Your Data**:
   - Check table structure matches expected schema
   - Verify data quality and completeness
   - Note any missing columns or different naming

### After Database Access
1. **Run Full Training Pipeline**:
   ```bash
   python train_learn2rank_models.py
   ```

2. **Analyze Results**:
   - Review `training_results.csv` for performance comparison
   - Check individual prediction files in result folders
   - Compare algorithm performance across time periods

3. **Production Deployment**:
   - Select best-performing algorithm
   - Set up automated retraining schedule
   - Implement real-time prediction system

## Troubleshooting

### MySQL Connection Issues
- Ensure MySQL service is running
- Check firewall settings
- Verify credentials and database name
- Try connecting with MySQL Workbench first

### Missing Dependencies
```bash
pip install mysql-connector-python xgboost torch pandas numpy tqdm
```

### Performance Issues
- Reduce training window size for faster training
- Use GPU acceleration for neural networks if available
- Sample data for initial testing

## Results Interpretation
- **Positive Long-Short Returns**: Model successfully identifies outperforming stocks
- **High Win Rate**: Consistent performance across time periods
- **NDCG > 0.5**: Good ranking quality
- **Compare Algorithms**: Choose based on your specific requirements (speed vs accuracy)

## Contact & Support
For questions about the implementation or database integration, refer to the individual script files which contain detailed comments and error handling.
