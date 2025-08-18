"""
Unit tests for backtesting metrics.
"""

import pytest
import numpy as np

from src.backtesting.metrics import (
    calculate_mse, calculate_mae, calculate_mape, calculate_rmse,
    calculate_correlation, calculate_directional_accuracy,
    calculate_confidence_calibration, create_performance_summary
)


class TestBacktestingMetrics:
    """Test backtesting metrics calculations."""
    
    def setup_method(self):
        """Setup test data."""
        self.actual = [100.0, 105.0, 102.0, 108.0, 110.0]
        self.predicted = [101.0, 104.0, 103.0, 107.0, 109.0]
        self.previous_actual = [98.0, 100.0, 105.0, 102.0, 108.0]
        self.confidence = [0.8, 0.9, 0.7, 0.85, 0.75]
    
    def test_calculate_mse(self):
        """Test Mean Squared Error calculation."""
        mse = calculate_mse(self.actual, self.predicted)
        
        # Manual calculation: [(100-101)² + (105-104)² + (102-103)² + (108-107)² + (110-109)²] / 5
        # = [1 + 1 + 1 + 1 + 1] / 5 = 1.0
        expected = 1.0
        assert abs(mse - expected) < 0.0001
    
    def test_calculate_mae(self):
        """Test Mean Absolute Error calculation."""
        mae = calculate_mae(self.actual, self.predicted)
        
        # Manual calculation: [|100-101| + |105-104| + |102-103| + |108-107| + |110-109|] / 5
        # = [1 + 1 + 1 + 1 + 1] / 5 = 1.0
        expected = 1.0
        assert abs(mae - expected) < 0.0001
    
    def test_calculate_mape(self):
        """Test Mean Absolute Percentage Error calculation."""
        mape = calculate_mape(self.actual, self.predicted)
        
        # Manual calculation: [|100-101|/100 + |105-104|/105 + ...] * 100 / 5
        expected_errors = [1/100, 1/105, 1/102, 1/108, 1/110]
        expected = np.mean(expected_errors) * 100
        
        assert abs(mape - expected) < 0.01
    
    def test_calculate_rmse(self):
        """Test Root Mean Squared Error calculation."""
        rmse = calculate_rmse(self.actual, self.predicted)
        
        # RMSE = sqrt(MSE) = sqrt(1.0) = 1.0
        expected = 1.0
        assert abs(rmse - expected) < 0.0001
    
    def test_calculate_correlation(self):
        """Test correlation calculation."""
        correlation = calculate_correlation(self.actual, self.predicted)
        
        # Should be high positive correlation since predicted closely follows actual
        assert correlation > 0.9
        assert correlation <= 1.0
    
    def test_calculate_directional_accuracy(self):
        """Test directional accuracy calculation."""
        accuracy = calculate_directional_accuracy(
            self.actual, self.predicted, self.previous_actual
        )
        
        # Check directions manually:
        # Actual: up, down, up, up (4 moves)
        # Predicted: up, down, up, up (4 moves)
        # All directions match, so accuracy should be 100%
        expected = 100.0
        assert abs(accuracy - expected) < 0.1
    
    def test_calculate_confidence_calibration(self):
        """Test confidence calibration calculation."""
        calibration = calculate_confidence_calibration(
            self.actual, self.predicted, self.confidence, num_bins=5
        )
        
        assert "bin_centers" in calibration
        assert "bin_accuracies" in calibration
        assert "bin_counts" in calibration
        
        assert len(calibration["bin_centers"]) == 5
        assert len(calibration["bin_accuracies"]) == 5
        assert len(calibration["bin_counts"]) == 5
        
        # Total count should equal number of predictions
        assert sum(calibration["bin_counts"]) == len(self.actual)
    
    def test_create_performance_summary(self):
        """Test creating comprehensive performance summary."""
        summary = create_performance_summary(
            self.actual,
            self.predicted,
            confidence_scores=self.confidence,
            previous_actual=self.previous_actual
        )
        
        # Check that all expected sections are present
        assert "basic_metrics" in summary
        assert "error_distribution" in summary
        assert "directional_accuracy" in summary
        assert "confidence_calibration" in summary
        
        # Check basic metrics
        basic = summary["basic_metrics"]
        assert basic["count"] == 5
        assert basic["mse"] == 1.0
        assert basic["mae"] == 1.0
        assert basic["rmse"] == 1.0
        
        # Check error distribution
        err_dist = summary["error_distribution"]
        assert "mean_error" in err_dist
        assert "std_error" in err_dist
        assert "median_error" in err_dist
        
        # Check directional accuracy
        assert summary["directional_accuracy"] == 100.0
    
    def test_mismatched_lengths(self):
        """Test that mismatched input lengths raise errors."""
        actual_short = [100.0, 105.0]
        predicted_long = [101.0, 104.0, 103.0]
        
        with pytest.raises(ValueError, match="must have same length"):
            calculate_mse(actual_short, predicted_long)
        
        with pytest.raises(ValueError, match="must have same length"):
            calculate_mae(actual_short, predicted_long)
        
        with pytest.raises(ValueError, match="must have same length"):
            calculate_correlation(actual_short, predicted_long)
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Single value
        single_actual = [100.0]
        single_predicted = [101.0]
        
        mse = calculate_mse(single_actual, single_predicted)
        assert mse == 1.0
        
        mae = calculate_mae(single_actual, single_predicted)
        assert mae == 1.0
        
        # Perfect predictions
        perfect_actual = [100.0, 105.0, 102.0]
        perfect_predicted = [100.0, 105.0, 102.0]
        
        mse = calculate_mse(perfect_actual, perfect_predicted)
        assert mse == 0.0
        
        mae = calculate_mae(perfect_actual, perfect_predicted)
        assert mae == 0.0
        
        correlation = calculate_correlation(perfect_actual, perfect_predicted)
        assert abs(correlation - 1.0) < 0.0001


if __name__ == "__main__":
    pytest.main([__file__])
