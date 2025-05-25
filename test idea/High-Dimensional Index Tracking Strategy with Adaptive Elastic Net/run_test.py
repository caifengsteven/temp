from test_strategy_with_simulated_data import test_with_simulated_data

if __name__ == "__main__":
    # Test with simulated data
    test_with_simulated_data(
        num_assets=100,
        num_days=1000,
        lambda1=1e-5,
        lambda2=1e-3,
        lambda_c=5e-5,  # Small turnover penalty
        tau=1,
        lookback_window=250,
        rebalance_period=21
    )
