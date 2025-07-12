#!/usr/bin/env python3
"""
Quick test of synthetic data generation
"""

import sys
import os
import importlib.util

def test_synthetic_data():
    """Test synthetic data generation"""
    
    # Import the strategy module
    spec = importlib.util.spec_from_file_location(
        "strategy", 
        "2410.23294v1_test_strategy.py"
    )
    strategy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strategy)
    
    print("Testing synthetic data generation...")
    
    # Test data generation
    data = strategy.get_forex_data(
        use_bloomberg=False,
        start_date='2022-01-01',
        end_date='2022-01-31'
    )
    
    print(f"✓ Generated {len(data)} data points")
    print(f"✓ Columns: {list(data.columns)}")
    print(f"✓ Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"✓ Price range: {data['mid_price'].min():.4f} to {data['mid_price'].max():.4f}")
    
    # Test environment creation
    env = strategy.ForexTradingEnvironment(
        data.head(1000), 
        action_type="continuous", 
        persistence=1
    )
    
    state = env.reset()
    print(f"✓ Environment created, state dimension: {len(state)}")
    
    # Test agent creation
    agent = strategy.FittedNaturalActorCritic(
        state_dim=len(state),
        action_type="continuous",
        action_dim=1
    )
    
    print("✓ Agent created successfully")
    
    # Test one step
    action = agent.select_action(state)
    next_state, reward, done, info = env.step(action)
    
    print(f"✓ Environment step successful")
    print(f"  Action: {action:.3f}")
    print(f"  Reward: {reward:.3f}")
    print(f"  Done: {done}")
    
    print("\n✓ All basic functionality tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_synthetic_data()
        print("\n🎉 Synthetic data mode is working correctly!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
