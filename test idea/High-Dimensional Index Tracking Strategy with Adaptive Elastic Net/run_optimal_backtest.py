from test_strategy_with_simulated_data import test_with_simulated_data

if __name__ == "__main__":
    # Run backtest with optimal parameters
    # Based on parameter sensitivity test, the best parameters are:
    # lambda1=1e-05, lambda2=0.01, lambda_c=0.0001
    
    test_with_simulated_data(
        num_assets=100,
        num_days=1000,
        lambda1=1e-5,
        lambda2=1e-2,
        lambda_c=1e-4,
        tau=1,
        lookback_window=250,
        rebalance_period=21
    )
