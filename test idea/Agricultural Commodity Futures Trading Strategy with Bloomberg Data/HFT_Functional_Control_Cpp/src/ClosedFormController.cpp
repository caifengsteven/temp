#include "../include/ClosedFormController.h"
#include <cmath>

ClosedFormController::ClosedFormController(std::shared_ptr<OptimalExecutionSimulator> simulator)
    : simulator(simulator) {
    
    // Precompute h1 and h2 functions
    precomputeHFunctions();
}

void ClosedFormController::precomputeHFunctions() {
    // Parameters
    int T = simulator->getT();
    double dt = simulator->getDt();
    double alpha = simulator->getAlpha();
    double kappa = simulator->getKappa();
    double A = simulator->getA();
    double phi = simulator->getPhi();
    
    // Initialize h1 and h2 at terminal time T
    h1.resize(T + 1, 0.0);
    h2.resize(T + 1, 0.0);
    h1[T] = 0.0;
    h2[T] = -2.0 * A;
    
    // Solve ODEs backwards in time
    for (int i = T - 1; i >= 0; --i) {
        // Update h2 using Euler method
        double h2_dot = -(2.0 * phi - alpha * alpha / (2.0 * kappa)) - 
                        (alpha / kappa) * h2[i + 1] - 
                        (h2[i + 1] * h2[i + 1]) / (2.0 * kappa);
        h2[i] = h2[i + 1] - dt * h2_dot;
        
        // Update h1 using Euler method
        double h1_dot = -(alpha + h2[i + 1]) / (2.0 * kappa) * h1[i + 1];
        h1[i] = h1[i + 1] - dt * h1_dot;
    }
}

double ClosedFormController::control(const std::vector<double>& state) {
    double remaining_time = state[0];
    double inventory = state[1];
    
    // Get time index
    int t = simulator->getT() - static_cast<int>(remaining_time);
    
    // Compute control
    double alpha = simulator->getAlpha();
    double kappa = simulator->getKappa();
    
    double trading_rate = h1[t] / (2.0 * kappa) + (alpha + h2[t]) / (2.0 * kappa) * inventory;
    
    return trading_rate;
}
