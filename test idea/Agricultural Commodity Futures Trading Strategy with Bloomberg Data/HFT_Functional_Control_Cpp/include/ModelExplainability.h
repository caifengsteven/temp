#pragma once

#include "OptimalExecutionSimulator.h"
#include "NeuralNetworkController.h"
#include <vector>
#include <tuple>
#include <memory>

/**
 * @brief Project controls from the neural network onto the closed-form manifold
 * @param controller Controller to project
 * @param simulator Simulator for context
 * @param time_points Time points to evaluate
 * @return Tuple of (h1_tilde, h2_tilde, r_squared)
 */
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> 
projectControlsOnClosedForm(
    std::shared_ptr<NeuralNetworkController> controller,
    std::shared_ptr<OptimalExecutionSimulator> simulator,
    const std::vector<int>& time_points
);
