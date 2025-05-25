#pragma once

#include "OptimalExecutionSimulator.h"
#include "TradingNetwork.h"
#include <torch/torch.h>
#include <memory>
#include <vector>
#include <string>

/**
 * @class NeuralNetworkController
 * @brief Neural network controller for optimal execution
 */
class NeuralNetworkController {
public:
    /**
     * @brief Constructor
     * @param simulator Simulator for the optimal execution problem
     * @param multi_preference Whether to include risk aversion parameters as inputs
     * @param hidden_dims Dimensions of hidden layers
     * @param learning_rate Learning rate for optimization
     * @param batch_size Batch size for training
     * @param device Device to use for training ('cpu' or 'cuda')
     */
    NeuralNetworkController(std::shared_ptr<OptimalExecutionSimulator> simulator,
                           bool multi_preference = false,
                           const std::vector<int>& hidden_dims = {5, 5, 5},
                           double learning_rate = 5e-4,
                           int batch_size = 64,
                           const std::string& device = "cpu");

    /**
     * @brief Return trading rate for a given state
     * @param state State of the system (T-t, q)
     * @param A Terminal inventory penalty (only used in multi-preference mode)
     * @param phi Running inventory penalty (only used in multi-preference mode)
     * @return Trading rate for the given state
     */
    double control(const std::vector<double>& state, double A = 0.0, double phi = 0.0);

    /**
     * @brief Train the controller on simulated data
     * @param n_iterations Number of SGD iterations
     * @param n_validation Number of validation samples
     * @param initial_inventory Initial inventory
     * @param tile_size Number of inventory samples per Brownian path
     * @param verbose Whether to print progress
     * @return Dictionary containing training history
     */
    struct TrainingHistory {
        std::vector<double> train_loss;
        std::vector<double> val_loss;
    };

    TrainingHistory trainOnSimulatedData(int n_iterations = 100000,
                                        int n_validation = 1000,
                                        double initial_inventory = -100.0,
                                        int tile_size = 3,
                                        bool verbose = true);

    /**
     * @brief Save the model to a file
     * @param path Path to save the model
     */
    void saveModel(const std::string& path);

    /**
     * @brief Load the model from a file
     * @param path Path to load the model from
     */
    void loadModel(const std::string& path);

private:
    /**
     * @brief Perform a single training step
     * @param initial_inventory Initial inventory
     * @param tile_size Number of inventory samples per Brownian path
     * @return Training loss
     */
    double trainStep(double initial_inventory, int tile_size);

    /**
     * @brief Validate the model
     * @param n_samples Number of validation samples
     * @param initial_inventory Initial inventory
     * @return Validation loss
     */
    double validate(int n_samples, double initial_inventory);

    std::shared_ptr<OptimalExecutionSimulator> simulator;
    std::shared_ptr<TradingNetwork> model;
    torch::optim::Adam optimizer;
    int batch_size;
    std::string device;
};
