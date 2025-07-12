#!/usr/bin/env python3
"""
Example usage of the Bloomberg Forex Trading Strategy
This script demonstrates different ways to use the enhanced FNAC implementation
"""

import sys
import os

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def example_basic_usage():
    """Example 1: Basic usage with default settings"""
    print("Example 1: Basic Usage")
    print("=" * 40)
    
    from config import get_config
    
    # Load default configuration
    config = get_config()
    
    # Modify for quick demo
    config['training']['episodes'] = 10  # Reduced for demo
    config['data']['start_date'] = '2022-01-01'
    config['data']['end_date'] = '2022-06-30'
    
    print("Configuration loaded:")
    print(f"  Currency: {config['data']['currency_pair']}")
    print(f"  Episodes: {config['training']['episodes']}")
    print(f"  Bloomberg: {config['data']['use_bloomberg']}")

def example_synthetic_data():
    """Example 2: Force synthetic data usage"""
    print("\nExample 2: Synthetic Data Mode")
    print("=" * 40)
    
    # Import the main functions
    from config import get_config
    
    config = get_config()
    
    # Force synthetic data
    config['data']['use_bloomberg'] = False
    config['training']['episodes'] = 5  # Very quick demo
    
    print("Configured for synthetic data:")
    print(f"  Bloomberg disabled: {not config['data']['use_bloomberg']}")
    print(f"  Quick training: {config['training']['episodes']} episodes")

def example_custom_currency():
    """Example 3: Custom currency pair"""
    print("\nExample 3: Custom Currency Pair")
    print("=" * 40)
    
    from config import get_config
    
    config = get_config()
    
    # Change to GBP/USD
    config['data']['currency_pair'] = 'GBPUSD Curncy'
    config['training']['episodes'] = 15
    
    print("Configured for GBP/USD:")
    print(f"  Currency: {config['data']['currency_pair']}")
    print(f"  Training episodes: {config['training']['episodes']}")

def example_risk_averse():
    """Example 4: Risk-averse configuration"""
    print("\nExample 4: Risk-Averse Settings")
    print("=" * 40)
    
    from config import get_config
    
    config = get_config()
    
    # Enable risk-averse features
    config['risk']['test_risk_averse'] = True
    config['risk']['mean_volatility_lambda'] = 0.005  # Higher risk aversion
    config['risk']['rcvar_rho'] = 0.3  # More conservative
    
    print("Risk-averse configuration:")
    print(f"  Mean-Vol λ: {config['risk']['mean_volatility_lambda']}")
    print(f"  RCVaR ρ: {config['risk']['rcvar_rho']}")

def example_model_training():
    """Example 5: Train and save a model"""
    print("\nExample 5: Model Training and Saving")
    print("=" * 40)
    
    try:
        # Import our strategy components
        from config import get_config
        
        config = get_config()
        
        # Quick training setup
        config['data']['use_bloomberg'] = False  # Use synthetic for demo
        config['training']['episodes'] = 3  # Very quick
        config['output']['save_models'] = True
        
        print("Model training configuration:")
        print(f"  Data source: {'Bloomberg' if config['data']['use_bloomberg'] else 'Synthetic'}")
        print(f"  Episodes: {config['training']['episodes']}")
        print(f"  Save models: {config['output']['save_models']}")
        
        # Note: Actual training would be done by importing and running the main strategy
        print("  (To actually train, run the main strategy file)")
        
    except ImportError as e:
        print(f"Import error: {e}")

def example_bloomberg_test():
    """Example 6: Test Bloomberg connection"""
    print("\nExample 6: Bloomberg Connection Test")
    print("=" * 40)
    
    try:
        # Test if Bloomberg is available
        import xbbg
        print("✓ xbbg library available")
        
        # You could run a quick test here
        print("  To test connection, run: python test_bloomberg_connection.py")
        
    except ImportError:
        print("✗ xbbg library not available")
        print("  Install with: pip install xbbg")
        print("  Or use synthetic data mode")

def example_multi_currency():
    """Example 7: Multi-currency testing setup"""
    print("\nExample 7: Multi-Currency Testing")
    print("=" * 40)
    
    from config import get_config
    
    config = get_config()
    
    # Show multi-currency configuration
    pairs = config['multi_currency']['currency_pairs']
    episodes = config['multi_currency']['quick_test_episodes']
    
    print("Multi-currency configuration:")
    print(f"  Currency pairs: {len(pairs)}")
    for pair in pairs:
        print(f"    - {pair}")
    print(f"  Episodes per pair: {episodes}")
    print("  To run: python 2410.23294v1_test_strategy.py --multi-currency")

def main():
    """Run all examples"""
    print("Bloomberg Forex Trading Strategy - Usage Examples")
    print("=" * 60)
    
    examples = [
        example_basic_usage,
        example_synthetic_data,
        example_custom_currency,
        example_risk_averse,
        example_model_training,
        example_bloomberg_test,
        example_multi_currency
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Error in {example_func.__name__}: {e}")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Test Bloomberg connection:")
    print("   python test_bloomberg_connection.py")
    print()
    print("2. Run basic strategy:")
    print("   python 2410.23294v1_test_strategy.py")
    print()
    print("3. Test multiple currencies:")
    print("   python 2410.23294v1_test_strategy.py --multi-currency")
    print()
    print("4. Customize settings:")
    print("   Edit config.py to modify parameters")
    print()
    print("5. Check requirements:")
    print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()
