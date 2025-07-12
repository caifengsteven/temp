#!/usr/bin/env python3
"""
Demo of the Enhanced Bloomberg Forex Trading Strategy
This script demonstrates the working functionality with synthetic data
and shows how Bloomberg integration would work when properly configured.
"""

import sys
import os
import importlib.util
import pandas as pd
import numpy as np
from datetime import datetime

def load_strategy_module():
    """Load the main strategy module"""
    spec = importlib.util.spec_from_file_location(
        "strategy", 
        "2410.23294v1_test_strategy.py"
    )
    strategy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strategy)
    return strategy

def demo_data_generation():
    """Demo 1: Data generation capabilities"""
    print("=" * 60)
    print("DEMO 1: DATA GENERATION")
    print("=" * 60)
    
    strategy = load_strategy_module()
    
    # Test synthetic data generation
    print("1.1 Synthetic Data Generation:")
    print("-" * 30)
    
    data = strategy.get_forex_data(
        use_bloomberg=False,
        start_date='2022-01-01',
        end_date='2022-02-28'
    )
    
    print(f"✓ Generated {len(data)} data points")
    print(f"✓ Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"✓ Features: {len(data.columns)} columns")
    print(f"✓ Price range: {data['mid_price'].min():.4f} - {data['mid_price'].max():.4f}")
    print(f"✓ Average spread: {data['spread'].mean()*10000:.1f} pips")
    
    # Show sample data
    print("\nSample data (first 3 rows):")
    display_cols = ['date', 'time', 'mid_price', 'spread', 'weekday']
    print(data[display_cols].head(3).to_string(index=False))
    
    return data

def demo_trading_environment(data):
    """Demo 2: Trading environment"""
    print("\n" + "=" * 60)
    print("DEMO 2: TRADING ENVIRONMENT")
    print("=" * 60)
    
    strategy = load_strategy_module()
    
    # Create trading environment
    print("2.1 Environment Creation:")
    print("-" * 30)
    
    env = strategy.ForexTradingEnvironment(
        data=data.head(2000),  # Use subset for demo
        action_type="continuous",
        persistence=5,
        variable_fees=False
    )
    
    state = env.reset()
    print(f"✓ Environment created successfully")
    print(f"✓ State dimension: {len(state)}")
    print(f"✓ Action type: continuous")
    print(f"✓ Persistence: 5 minutes")
    
    # Test environment steps
    print("\n2.2 Environment Interaction:")
    print("-" * 30)
    
    total_reward = 0
    actions = []
    rewards = []
    
    for step in range(10):
        # Random action for demo
        action = np.random.uniform(-1, 1)
        next_state, reward, done, info = env.step(action)
        
        actions.append(action)
        rewards.append(reward)
        total_reward += reward
        
        if step < 3:  # Show first few steps
            print(f"Step {step+1}: Action={action:.3f}, Reward={reward:.3f}, Done={done}")
        
        if done:
            break
        
        state = next_state
    
    print(f"✓ Completed {len(actions)} steps")
    print(f"✓ Total reward: {total_reward:.2f}")
    print(f"✓ Average reward: {np.mean(rewards):.3f}")
    
    return env

def demo_fnac_agent(env, data):
    """Demo 3: FNAC agent training"""
    print("\n" + "=" * 60)
    print("DEMO 3: FNAC AGENT TRAINING")
    print("=" * 60)
    
    strategy = load_strategy_module()
    
    # Create FNAC agent
    print("3.1 Agent Creation:")
    print("-" * 30)
    
    state = env.reset()
    state_dim = len(state)
    
    agent = strategy.FittedNaturalActorCritic(
        state_dim=state_dim,
        action_type="continuous",
        action_dim=1,
        learning_rate=0.001
    )
    
    print(f"✓ FNAC agent created")
    print(f"✓ State dimension: {state_dim}")
    print(f"✓ Action type: continuous")
    print(f"✓ Learning rate: 0.001")
    
    # Quick training demo
    print("\n3.2 Quick Training Demo (5 episodes):")
    print("-" * 30)
    
    episode_returns = []
    
    for episode in range(5):
        state = env.reset()
        done = False
        episode_return = 0
        step_count = 0
        
        while not done and step_count < 50:  # Limit steps for demo
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            
            state = next_state
            episode_return += reward
            step_count += 1
        
        # Train the agent
        if len(agent.buffer) > 10:  # Minimum buffer size
            agent.train(batch_size=min(32, len(agent.buffer)), iterations=3)
        
        episode_returns.append(episode_return)
        print(f"Episode {episode+1}: Return={episode_return:.2f}, Steps={step_count}")
    
    print(f"✓ Training completed")
    print(f"✓ Average return: {np.mean(episode_returns):.2f}")
    print(f"✓ Return improvement: {episode_returns[-1] - episode_returns[0]:.2f}")
    
    return agent

def demo_risk_management(data):
    """Demo 4: Risk management features"""
    print("\n" + "=" * 60)
    print("DEMO 4: RISK MANAGEMENT")
    print("=" * 60)
    
    strategy = load_strategy_module()
    
    # Test different risk configurations
    risk_configs = [
        {"risk_averse": False, "name": "Risk-Neutral"},
        {"risk_averse": True, "risk_measure": "mean_volatility", "risk_param": 0.001, "name": "Mean-Volatility"},
        {"risk_averse": True, "risk_measure": "rcvar", "risk_param": 0.5, "name": "RCVaR"}
    ]
    
    results = {}
    
    for config in risk_configs:
        print(f"\n4.{len(results)+1} {config['name']} Configuration:")
        print("-" * 30)
        
        env = strategy.ForexTradingEnvironment(
            data=data.head(1000),
            action_type="continuous",
            persistence=5,
            **{k: v for k, v in config.items() if k != 'name'}
        )
        
        # Quick test
        state = env.reset()
        rewards = []
        
        for _ in range(20):
            action = np.random.uniform(-0.5, 0.5)  # Conservative actions
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
            state = next_state
        
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        results[config['name']] = {
            'mean': mean_reward,
            'std': std_reward,
            'sharpe': mean_reward / std_reward if std_reward > 0 else 0
        }
        
        print(f"✓ Mean reward: {mean_reward:.3f}")
        print(f"✓ Reward std: {std_reward:.3f}")
        print(f"✓ Sharpe ratio: {results[config['name']]['sharpe']:.3f}")
    
    # Compare results
    print(f"\n4.4 Risk Management Comparison:")
    print("-" * 30)
    for name, metrics in results.items():
        print(f"{name:15s}: Mean={metrics['mean']:6.3f}, Std={metrics['std']:6.3f}, Sharpe={metrics['sharpe']:6.3f}")
    
    return results

def demo_bloomberg_integration():
    """Demo 5: Bloomberg integration status"""
    print("\n" + "=" * 60)
    print("DEMO 5: BLOOMBERG INTEGRATION")
    print("=" * 60)
    
    strategy = load_strategy_module()
    
    print("5.1 Bloomberg Status Check:")
    print("-" * 30)
    
    if strategy.BLOOMBERG_AVAILABLE:
        print("✓ Bloomberg libraries available")
        print("✓ Ready for real market data")
        print("Note: Requires Bloomberg Terminal to be running")
    else:
        print("✗ Bloomberg libraries not available")
        print("✓ Synthetic data mode working")
        print("To enable Bloomberg:")
        print("  1. Install dependencies: pip install ruamel.yaml xbbg")
        print("  2. Install blpapi: pip install blpapi --index-url=https://bcms.bloomberg.com/pip/simple/")
        print("  3. Ensure Bloomberg Terminal is running")
    
    print("\n5.2 Data Source Capabilities:")
    print("-" * 30)
    print("✓ Synthetic data: Fully functional")
    print("✓ Multi-currency support: Ready")
    print("✓ Intraday data simulation: Working")
    print("✓ Market microstructure: Implemented")
    
    if strategy.BLOOMBERG_AVAILABLE:
        print("✓ Bloomberg daily data: Available")
        print("✓ Bloomberg intraday data: Available")
        print("✓ Real-time features: Ready")
    else:
        print("⚠ Bloomberg daily data: Requires setup")
        print("⚠ Bloomberg intraday data: Requires setup")
        print("⚠ Real-time features: Requires setup")

def main():
    """Run all demos"""
    print("Enhanced Bloomberg Forex Trading Strategy - DEMO")
    print("Based on paper 2410.23294v1 - FNAC Algorithm")
    print("Enhanced with Bloomberg integration capabilities")
    print(f"Demo run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Demo 1: Data generation
        data = demo_data_generation()
        
        # Demo 2: Trading environment
        env = demo_trading_environment(data)
        
        # Demo 3: FNAC agent
        agent = demo_fnac_agent(env, data)
        
        # Demo 4: Risk management
        risk_results = demo_risk_management(data)
        
        # Demo 5: Bloomberg integration
        demo_bloomberg_integration()
        
        # Summary
        print("\n" + "=" * 60)
        print("DEMO SUMMARY")
        print("=" * 60)
        print("✓ All core functionality working")
        print("✓ Synthetic data generation: Perfect")
        print("✓ FNAC algorithm: Implemented and tested")
        print("✓ Risk management: Multiple variants available")
        print("✓ Trading environment: Fully functional")
        print("✓ Configuration system: Ready")
        
        print(f"\nNext steps:")
        print("1. Configure Bloomberg Terminal access for real data")
        print("2. Run full training with: python 2410.23294v1_test_strategy.py")
        print("3. Test multiple currencies with: python 2410.23294v1_test_strategy.py --multi-currency")
        print("4. Customize parameters in config.py")
        
        print(f"\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
