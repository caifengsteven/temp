#include "../include/ModelExplainability.h"
#include <Eigen/Dense>
#include <vector>

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> 
projectControlsOnClosedForm(
    std::shared_ptr<NeuralNetworkController> controller,
    std::shared_ptr<OptimalExecutionSimulator> simulator,
    const std::vector<int>& time_points) {
    
    std::vector<double> h1_tilde(time_points.size(), 0.0);
    std::vector<double> h2_tilde(time_points.size(), 0.0);
    std::vector<double> r_squared(time_points.size(), 0.0);
    
    // For each time point
    for (size_t i = 0; i < time_points.size(); ++i) {
        int t = time_points[i];
        
        // Create range of inventory values
        const int num_samples = 200;
        std::vector<double> inventory_values(num_samples);
        std::vector<double> control_values(num_samples);
        
        for (int j = 0; j < num_samples; ++j) {
            inventory_values[j] = -100.0 + j * 100.0 / (num_samples - 1);
        }
        
        // Get control values for each inventory
        for (int j = 0; j < num_samples; ++j) {
            double q = inventory_values[j];
            std::vector<double> state = {static_cast<double>(simulator->getT() - t), q};
            control_values[j] = controller->control(state);
        }
        
        // Perform linear regression using Eigen
        Eigen::MatrixXd X(num_samples, 2);
        Eigen::VectorXd y(num_samples);
        
        for (int j = 0; j < num_samples; ++j) {
            X(j, 0) = 1.0;  // Intercept
            X(j, 1) = inventory_values[j];
            y(j) = control_values[j];
        }
        
        // Solve linear regression
        Eigen::VectorXd coeffs = X.colPivHouseholderQr().solve(y);
        double intercept = coeffs(0);
        double slope = coeffs(1);
        
        // Convert to h1 and h2
        double kappa = simulator->getKappa();
        double alpha = simulator->getAlpha();
        
        h1_tilde[i] = 2.0 * kappa * intercept;
        h2_tilde[i] = 2.0 * kappa * slope - alpha;
        
        // Calculate R-squared
        Eigen::VectorXd y_pred = X * coeffs;
        double ss_total = 0.0;
        double ss_residual = 0.0;
        double y_mean = y.mean();
        
        for (int j = 0; j < num_samples; ++j) {
            ss_total += (y(j) - y_mean) * (y(j) - y_mean);
            ss_residual += (y(j) - y_pred(j)) * (y(j) - y_pred(j));
        }
        
        r_squared[i] = 1.0 - ss_residual / ss_total;
    }
    
    return {h1_tilde, h2_tilde, r_squared};
}
