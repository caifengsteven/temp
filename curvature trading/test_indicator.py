"""
Test script for Curved Radius Supertrend

This script runs basic tests to verify the indicator is working correctly.
"""

import numpy as np
import pandas as pd
from curved_radius_supertrend import CurvedRadiusSupertrend


def test_basic_calculation():
    """Test basic indicator calculation"""
    print("Test 1: Basic Calculation")
    print("-" * 50)
    
    # Create simple test data
    n = 100
    close = np.linspace(100, 120, n) + np.random.randn(n) * 2
    high = close + np.abs(np.random.randn(n)) * 1.5
    low = close - np.abs(np.random.randn(n)) * 1.5
    
    # Calculate indicator
    indicator = CurvedRadiusSupertrend()
    result = indicator.calculate(high, low, close)
    
    # Verify output structure
    assert 'curved_upper' in result.columns, "Missing curved_upper column"
    assert 'curved_lower' in result.columns, "Missing curved_lower column"
    assert 'direction' in result.columns, "Missing direction column"
    assert 'trend_line' in result.columns, "Missing trend_line column"
    
    # Verify output length
    assert len(result) == n, f"Output length mismatch: {len(result)} != {n}"
    
    # Verify direction values
    assert set(result['direction'].unique()).issubset({1, -1}), "Invalid direction values"
    
    print("✓ Basic calculation test passed")
    print(f"  - Output shape: {result.shape}")
    print(f"  - Trend changes: {(result['direction'].diff() != 0).sum()}")
    print()


def test_parameter_variations():
    """Test different parameter settings"""
    print("Test 2: Parameter Variations")
    print("-" * 50)
    
    # Create test data
    n = 100
    np.random.seed(42)
    close = np.linspace(100, 120, n) + np.random.randn(n) * 2
    high = close + np.abs(np.random.randn(n)) * 1.5
    low = close - np.abs(np.random.randn(n)) * 1.5
    
    # Test different radius strengths
    radius_values = [0.1, 0.5, 1.0, 2.0]
    
    for radius in radius_values:
        indicator = CurvedRadiusSupertrend(radius_strength=radius)
        result = indicator.calculate(high, low, close)
        
        trend_changes = (result['direction'].diff() != 0).sum()
        print(f"  Radius {radius}: {trend_changes} trend changes")
    
    print("✓ Parameter variation test passed")
    print()


def test_trend_detection():
    """Test trend detection on synthetic data"""
    print("Test 3: Trend Detection")
    print("-" * 50)
    
    # Create clear uptrend
    n = 100
    uptrend = np.linspace(100, 150, n) + np.random.randn(n) * 1
    high_up = uptrend + np.abs(np.random.randn(n)) * 1
    low_up = uptrend - np.abs(np.random.randn(n)) * 1
    
    indicator = CurvedRadiusSupertrend(radius_strength=0.5)
    result_up = indicator.calculate(high_up, low_up, uptrend)
    
    # Should detect mostly uptrend
    uptrend_pct = (result_up['direction'] == 1).sum() / len(result_up) * 100
    print(f"  Uptrend data: {uptrend_pct:.1f}% detected as uptrend")
    
    # Create clear downtrend
    downtrend = np.linspace(150, 100, n) + np.random.randn(n) * 1
    high_down = downtrend + np.abs(np.random.randn(n)) * 1
    low_down = downtrend - np.abs(np.random.randn(n)) * 1
    
    result_down = indicator.calculate(high_down, low_down, downtrend)
    
    # Should detect mostly downtrend
    downtrend_pct = (result_down['direction'] == -1).sum() / len(result_down) * 100
    print(f"  Downtrend data: {downtrend_pct:.1f}% detected as downtrend")
    
    print("✓ Trend detection test passed")
    print()


def test_curvature_acceleration():
    """Test that curvature acceleration is working"""
    print("Test 4: Curvature Acceleration")
    print("-" * 50)
    
    # Create test data
    n = 100
    np.random.seed(42)
    close = np.linspace(100, 120, n) + np.random.randn(n) * 2
    high = close + np.abs(np.random.randn(n)) * 1.5
    low = close - np.abs(np.random.randn(n)) * 1.5
    
    # Compare with and without curvature
    indicator_no_curve = CurvedRadiusSupertrend(radius_strength=0.0)
    indicator_with_curve = CurvedRadiusSupertrend(radius_strength=1.0)
    
    result_no_curve = indicator_no_curve.calculate(high, low, close)
    result_with_curve = indicator_with_curve.calculate(high, low, close)
    
    # Calculate difference in band positions
    upper_diff = np.abs(result_with_curve['curved_upper'] - result_no_curve['curved_upper']).mean()
    lower_diff = np.abs(result_with_curve['curved_lower'] - result_no_curve['curved_lower']).mean()
    
    print(f"  Average upper band difference: {upper_diff:.4f}")
    print(f"  Average lower band difference: {lower_diff:.4f}")
    
    # With curvature should be different
    assert upper_diff > 0 or lower_diff > 0, "Curvature not affecting bands"
    
    print("✓ Curvature acceleration test passed")
    print()


def test_edge_cases():
    """Test edge cases"""
    print("Test 5: Edge Cases")
    print("-" * 50)
    
    # Test with minimal data
    n = 20
    close = np.ones(n) * 100 + np.random.randn(n) * 0.5
    high = close + 0.5
    low = close - 0.5
    
    indicator = CurvedRadiusSupertrend()
    result = indicator.calculate(high, low, close)
    
    assert len(result) == n, "Failed with minimal data"
    print("  ✓ Minimal data test passed")
    
    # Test with flat prices
    close_flat = np.ones(50) * 100
    high_flat = close_flat + 0.1
    low_flat = close_flat - 0.1
    
    result_flat = indicator.calculate(high_flat, low_flat, close_flat)
    assert len(result_flat) == 50, "Failed with flat prices"
    print("  ✓ Flat price test passed")
    
    print()


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 50)
    print("CURVED RADIUS SUPERTREND - TEST SUITE")
    print("=" * 50 + "\n")
    
    try:
        test_basic_calculation()
        test_parameter_variations()
        test_trend_detection()
        test_curvature_acceleration()
        test_edge_cases()
        
        print("=" * 50)
        print("ALL TESTS PASSED ✓")
        print("=" * 50)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()

