#include "../include/NeuralNetworkController.h"
#include <iostream>
#include <random>
#include <chrono>

NeuralNetworkController::NeuralNetworkController(
    std::shared_ptr<OptimalExecutionSimulator> simulator,
    bool multi_preference,
    const std::vector<int>& hidden_dims,
    double learning_rate,
    int batch_size,
    const std::string& device)
    : simulator(simulator),
      batch_size(batch_size),
      device(device) {
    
    // Create neural network
    model = std::make_shared<TradingNetwork>(
        2,  // input_dim (t and q)
        hidden_dims,
        multi_preference
    );
    
    // Move model to device
    if (device == "cuda" && torch::cuda::is_available()) {
        model->to(torch::kCUDA);
    }
    
    // Create optimizer
    optimizer = torch::optim::Adam(model->parameters(), learning_rate);
}

double NeuralNetworkController::control(
    const std::vector<double>& state, double A, double phi) {
    
    // Convert to tensor
    torch::Tensor state_tensor;
    
    if (model->isMultiPreference()) {
        if (A == 0.0) A = simulator->getA();
        if (phi == 0.0) phi = simulator->getPhi();
        
        std::vector<double> extended_state = state;
        extended_state.push_back(A);
        extended_state.push_back(phi);
        
        state_tensor = torch::tensor(extended_state, torch::kFloat32);
    } else {
        state_tensor = torch::tensor(state, torch::kFloat32);
    }
    
    // Move to device
    if (device == "cuda" && torch::cuda::is_available()) {
        state_tensor = state_tensor.to(torch::kCUDA);
    }
    
    // Get prediction
    model->eval();
    torch::NoGradGuard no_grad;
    torch::Tensor output = model->forward(state_tensor);
    
    return output.item<double>();
}

NeuralNetworkController::TrainingHistory 
NeuralNetworkController::trainOnSimulatedData(
    int n_iterations, int n_validation, double initial_inventory,
    int tile_size, bool verbose) {
    
    // Training history
    TrainingHistory history;
    
    for (int i = 0; i < n_iterations; ++i) {
        // Training step
        double loss = trainStep(initial_inventory, tile_size);
        history.train_loss.push_back(loss);
        
        // Validation step
        if ((i + 1) % 100 == 0) {
            double val_loss = validate(n_validation, initial_inventory);
            history.val_loss.push_back(val_loss);
            
            if (verbose) {
                std::cout << "Iteration " << (i + 1) << "/" << n_iterations
                          << ", Loss: " << loss << ", Val Loss: " << val_loss << std::endl;
            }
        }
    }
    
    return history;
}

double NeuralNetworkController::trainStep(double initial_inventory, int tile_size) {
    model->train();
    optimizer.zero_grad();
    
    // Create mini-batch
    double total_loss = 0.0;
    
    // Random number generator for inventory sampling
    std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> A_dist(0.0001, 0.01);
    std::uniform_real_distribution<double> phi_dist(7e-5, 0.007);
    
    for (int b = 0; b < batch_size; ++b) {
        // Reset simulator with different random seed
        simulator->reset(initial_inventory);
        
        // Generate different initial inventories for same Brownian path
        std::vector<double> inventories(tile_size);
        for (int i = 0; i < tile_size; ++i) {
            inventories[i] = (0.8 + 0.4 * i / (tile_size - 1)) * initial_inventory;
        }
        
        for (double inventory : inventories) {
            // For multi-preference, sample different preferences
            if (model->isMultiPreference()) {
                // Sample A and phi from a reasonable range
                double A = A_dist(rng);
                double phi = phi_dist(rng);
                
                // Set simulator parameters
                simulator->setA(A);
                simulator->setPhi(phi);
            }
            
            // Reset simulator with this inventory
            auto state = simulator->reset(inventory);
            bool done = false;
            std::vector<double> episode_rewards;
            
            // Simulate trajectory
            while (!done) {
                // Get trading rate from current model
                torch::Tensor state_tensor;
                
                if (model->isMultiPreference()) {
                    std::vector<double> extended_state = state;
                    extended_state.push_back(simulator->getA());
                    extended_state.push_back(simulator->getPhi());
                    
                    state_tensor = torch::tensor(extended_state, torch::kFloat32);
                } else {
                    state_tensor = torch::tensor(state, torch::kFloat32);
                }
                
                // Move to device
                if (device == "cuda" && torch::cuda::is_available()) {
                    state_tensor = state_tensor.to(torch::kCUDA);
                }
                
                // Forward pass
                torch::Tensor trading_rate_tensor = model->forward(state_tensor);
                double trading_rate = trading_rate_tensor.item<double>();
                
                // Take action in simulator
                auto [next_state, reward, is_done, _] = simulator->step(trading_rate);
                state = next_state;
                episode_rewards.push_back(reward);
                done = is_done;
            }
            
            // Calculate total reward for this episode
            double episode_reward = 0.0;
            for (double r : episode_rewards) {
                episode_reward += r;
            }
            
            // Add negative reward as loss (we want to maximize reward)
            double loss = -episode_reward / (batch_size * tile_size);
            total_loss += loss;
        }
    }
    
    // Create tensor for backward pass
    torch::Tensor loss_tensor = torch::tensor(total_loss, torch::kFloat32);
    loss_tensor.requires_grad_(true);
    
    // Move to device
    if (device == "cuda" && torch::cuda::is_available()) {
        loss_tensor = loss_tensor.to(torch::kCUDA);
    }
    
    // Backward pass
    loss_tensor.backward();
    optimizer.step();
    
    return total_loss;
}

double NeuralNetworkController::validate(int n_samples, double initial_inventory) {
    model->eval();
    double val_loss = 0.0;
    
    // Random number generator for preference sampling
    std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> A_dist(0.0001, 0.01);
    std::uniform_real_distribution<double> phi_dist(7e-5, 0.007);
    
    torch::NoGradGuard no_grad;
    
    for (int i = 0; i < n_samples; ++i) {
        // For multi-preference, sample different preferences
        double A = 0.0, phi = 0.0;
        if (model->isMultiPreference()) {
            // Sample A and phi from range
            A = A_dist(rng);
            phi = phi_dist(rng);
            
            // Set simulator parameters
            simulator->setA(A);
            simulator->setPhi(phi);
        }
        
        // Define control function
        auto control_func = [this, A, phi](const std::vector<double>& state) {
            return this->control(state, A, phi);
        };
        
        // Simulate trajectory
        auto trajectory = simulator->simulateTrajectory(control_func, initial_inventory);
        
        val_loss -= trajectory.total_reward / n_samples;
    }
    
    return val_loss;
}

void NeuralNetworkController::saveModel(const std::string& path) {
    torch::save(model, path);
}

void NeuralNetworkController::loadModel(const std::string& path) {
    torch::load(model, path);
}
