#pragma once

#include "OptimalExecutionSimulator.h"
#include <vector>
#include <memory>

/**
 * @class ClosedFormController
 * @brief Closed-form controller based on PDE solution
 */
class ClosedFormController {
public:
    /**
     * @brief Constructor
     * @param simulator Simulator for the optimal execution problem
     */
    ClosedFormController(std::shared_ptr<OptimalExecutionSimulator> simulator);

    /**
     * @brief Return trading rate for a given state
     * @param state State of the system (T-t, q)
     * @return Trading rate for the given state
     */
    double control(const std::vector<double>& state);

    /**
     * @brief Get h1 function values
     * @return Vector of h1 values
     */
    const std::vector<double>& getH1() const { return h1; }

    /**
     * @brief Get h2 function values
     * @return Vector of h2 values
     */
    const std::vector<double>& getH2() const { return h2; }

private:
    /**
     * @brief Precompute h1 and h2 functions from ODEs
     */
    void precomputeHFunctions();

    std::shared_ptr<OptimalExecutionSimulator> simulator;
    std::vector<double> h1;
    std::vector<double> h2;
};
