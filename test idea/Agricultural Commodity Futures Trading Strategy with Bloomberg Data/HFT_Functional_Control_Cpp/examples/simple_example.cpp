#include "../include/OptimalExecutionSimulator.h"
#include "../include/ClosedFormController.h"
#include "../include/NeuralNetworkController.h"
#include <iostream>
#include <memory>
#include <vector>
#include <iomanip>

int main() {
    // Create simulator
    auto simulator = std::make_shared<OptimalExecutionSimulator>(
        77,  // T: 77 five-minute bins in a trading day
        1.0,  // dt
        100.0,  // initial_price
        0.1,  // vol
        0.1,  // alpha
        0.1,  // kappa
        0.01,  // A
        0.007,  // phi
        2.0,  // gamma
        false  // with_seasonality
    );
    
    // Create closed-form controller
    auto closed_form = std::make_shared<ClosedFormController>(simulator);
    
    // Determine device
    std::string device = torch::cuda::is_available() ? "cuda" : "cpu";
    std::cout << "Using device: " << device << std::endl;
    
    // Create neural network controller
    auto neural_net = std::make_shared<NeuralNetworkController>(
        simulator,
        false,  // multi_preference
        std::vector<int>{5, 5, 5},  // hidden_dims
        5e-4,  // learning_rate
        64,  // batch_size
        device
    );
    
    // Train the neural network (with fewer iterations for this example)
    std::cout << "Training neural network on simulated data..." << std::endl;
    auto history = neural_net->trainOnSimulatedData(
        1000,  // n_iterations (reduced for example)
        100,  // n_validation
        -100.0,  // initial_inventory
        3,  // tile_size
        true  // verbose
    );
    
    // Define control functions
    auto cf_control = [&closed_form](const std::vector<double>& state) {
        return closed_form->control(state);
    };
    
    auto nn_control = [&neural_net](const std::vector<double>& state) {
        return neural_net->control(state);
    };
    
    // Simulate trajectories
    double initial_inventory = -100.0;
    auto cf_traj = simulator->simulateTrajectory(cf_control, initial_inventory);
    auto nn_traj = simulator->simulateTrajectory(nn_control, initial_inventory);
    
    // Print results
    std::cout << "\nSimulation Results:" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    
    std::cout << "\nClosed-Form Controller:" << std::endl;
    std::cout << "  Final Inventory: " << cf_traj.inventory_path.back() << std::endl;
    std::cout << "  Total Reward: " << cf_traj.total_reward << std::endl;
    std::cout << "  Final Reward: " << cf_traj.final_reward << std::endl;
    
    std::cout << "\nNeural Network Controller:" << std::endl;
    std::cout << "  Final Inventory: " << nn_traj.inventory_path.back() << std::endl;
    std::cout << "  Total Reward: " << nn_traj.total_reward << std::endl;
    std::cout << "  Final Reward: " << nn_traj.final_reward << std::endl;
    
    // Print inventory path for both controllers
    std::cout << "\nInventory Path Comparison:" << std::endl;
    std::cout << "Time\tClosed-Form\tNeural Net" << std::endl;
    std::cout << "----\t----------\t---------" << std::endl;
    
    for (int i = 0; i < simulator->getT(); i += 10) {  // Print every 10th step
        std::cout << i << "\t" 
                  << cf_traj.inventory_path[i] << "\t\t" 
                  << nn_traj.inventory_path[i] << std::endl;
    }
    
    // Print final step
    std::cout << simulator->getT() << "\t" 
              << cf_traj.inventory_path.back() << "\t\t" 
              << nn_traj.inventory_path.back() << std::endl;
    
    return 0;
}
