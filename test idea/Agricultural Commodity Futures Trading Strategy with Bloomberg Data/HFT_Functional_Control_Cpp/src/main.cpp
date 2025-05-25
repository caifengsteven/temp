#include "../include/OptimalExecutionSimulator.h"
#include "../include/TradingNetwork.h"
#include "../include/NeuralNetworkController.h"
#include "../include/ClosedFormController.h"
#include "../include/ModelExplainability.h"

#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <torch/torch.h>

// For plotting
#include <matplotlibcpp.h>
namespace plt = matplotlibcpp;

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
    
    // Determine device
    std::string device = torch::cuda::is_available() ? "cuda" : "cpu";
    std::cout << "Using device: " << device << std::endl;
    
    // Create controllers
    auto closed_form = std::make_shared<ClosedFormController>(simulator);
    
    auto neural_net = std::make_shared<NeuralNetworkController>(
        simulator,
        false,  // multi_preference
        std::vector<int>{5, 5, 5},  // hidden_dims
        5e-4,  // learning_rate
        64,  // batch_size
        device
    );
    
    auto multi_pref_net = std::make_shared<NeuralNetworkController>(
        simulator,
        true,  // multi_preference
        std::vector<int>{5, 5, 5},  // hidden_dims
        5e-4,  // learning_rate
        64,  // batch_size
        device
    );
    
    // Train the neural network on simulated data
    std::cout << "Training neural network on simulated data..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto history = neural_net->trainOnSimulatedData(
        20000,  // n_iterations (reduced for demonstration)
        1000,  // n_validation
        -100.0,  // initial_inventory
        3,  // tile_size
        true  // verbose
    );
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    std::cout << "Training completed in " << duration << " seconds." << std::endl;
    
    // Save the trained model
    neural_net->saveModel("neural_net_model.pt");
    
    // Train the multi-preference neural network
    std::cout << "Training multi-preference neural network on simulated data..." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    
    auto multi_history = multi_pref_net->trainOnSimulatedData(
        20000,  // n_iterations (reduced for demonstration)
        1000,  // n_validation
        -100.0,  // initial_inventory
        3,  // tile_size
        true  // verbose
    );
    
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    std::cout << "Training completed in " << duration << " seconds." << std::endl;
    
    // Save the trained model
    multi_pref_net->saveModel("multi_pref_net_model.pt");
    
    // Create simulator with seasonality
    auto simulator_seasonal = std::make_shared<OptimalExecutionSimulator>(
        77,  // T
        1.0,  // dt
        100.0,  // initial_price
        0.1,  // vol
        0.1,  // alpha
        0.1,  // kappa
        0.01,  // A
        0.007,  // phi
        2.0,  // gamma
        true  // with_seasonality
    );
    
    // Train on simulator with seasonality
    auto neural_net_seasonal = std::make_shared<NeuralNetworkController>(
        simulator_seasonal,
        false,  // multi_preference
        std::vector<int>{5, 5, 5},  // hidden_dims
        5e-4,  // learning_rate
        64,  // batch_size
        device
    );
    
    std::cout << "Training neural network on simulated data with seasonality..." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    
    auto history_seasonal = neural_net_seasonal->trainOnSimulatedData(
        20000,  // n_iterations (reduced for demonstration)
        1000,  // n_validation
        -100.0,  // initial_inventory
        3,  // tile_size
        true  // verbose
    );
    
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    std::cout << "Training completed in " << duration << " seconds." << std::endl;
    
    // Save the trained model
    neural_net_seasonal->saveModel("neural_net_seasonal_model.pt");
    
    // Create sub-diffusive simulator (gamma=1.5)
    auto simulator_subdiff = std::make_shared<OptimalExecutionSimulator>(
        77,  // T
        1.0,  // dt
        100.0,  // initial_price
        0.1,  // vol
        0.1,  // alpha
        0.1,  // kappa
        0.5,  // A
        0.1,  // phi
        1.5,  // gamma
        false  // with_seasonality
    );
    
    // Train on sub-diffusive simulator
    auto neural_net_subdiff = std::make_shared<NeuralNetworkController>(
        simulator_subdiff,
        false,  // multi_preference
        std::vector<int>{5, 5, 5},  // hidden_dims
        5e-4,  // learning_rate
        64,  // batch_size
        device
    );
    
    std::cout << "Training neural network with sub-diffusive loss..." << std::endl;
    start_time = std::chrono::high_resolution_clock::now();
    
    auto history_subdiff = neural_net_subdiff->trainOnSimulatedData(
        20000,  // n_iterations (reduced for demonstration)
        1000,  // n_validation
        -100.0,  // initial_inventory
        3,  // tile_size
        true  // verbose
    );
    
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    std::cout << "Training completed in " << duration << " seconds." << std::endl;
    
    // Save the trained model
    neural_net_subdiff->saveModel("neural_net_subdiff_model.pt");
    
    // Evaluate and project each controller
    std::vector<int> time_points(simulator->getT() + 1);
    for (int i = 0; i <= simulator->getT(); ++i) {
        time_points[i] = i;
    }
    
    // Closed form values
    std::vector<double> h1_closed = closed_form->getH1();
    std::vector<double> h2_closed = closed_form->getH2();
    std::vector<double> r_squared_closed(time_points.size(), 1.0);
    
    // Project neural net controller
    auto [h1_nn, h2_nn, r_squared_nn] = projectControlsOnClosedForm(
        neural_net, simulator, time_points
    );
    
    // Project multi-preference neural net controller
    auto [h1_multi, h2_multi, r_squared_multi] = projectControlsOnClosedForm(
        multi_pref_net, simulator, time_points
    );
    
    // Project seasonal neural net controller
    auto [h1_seasonal, h2_seasonal, r_squared_seasonal] = projectControlsOnClosedForm(
        neural_net_seasonal, simulator_seasonal, time_points
    );
    
    // Project sub-diffusive neural net controller
    auto [h1_subdiff, h2_subdiff, r_squared_subdiff] = projectControlsOnClosedForm(
        neural_net_subdiff, simulator_subdiff, time_points
    );
    
    // Plot results using matplotlib-cpp
    plt::figure_size(1200, 900);
    
    // Plot h1 values
    plt::subplot(2, 2, 1);
    plt::plot(time_points, h1_closed, "k-", {{"label", "Closed-form solution"}, {"linewidth", "2"}});
    plt::plot(time_points, h1_nn, "b-", {{"label", "NN on simulations"}, {"linewidth", "2"}});
    plt::plot(time_points, h1_seasonal, "g--", {{"label", "NN with seasonality"}, {"linewidth", "2"}});
    plt::plot(time_points, h1_multi, "r-.", {{"label", "Multi-preference NN"}, {"linewidth", "2"}});
    plt::plot(time_points, h1_subdiff, "m:", {{"label", "NN with γ=3/2"}, {"linewidth", "2"}});
    plt::xlabel("Time");
    plt::ylabel("h1(t)");
    plt::legend();
    plt::grid(true);
    
    // Plot h2 values
    plt::subplot(2, 2, 2);
    plt::plot(time_points, h2_closed, "k-", {{"label", "Closed-form solution"}, {"linewidth", "2"}});
    plt::plot(time_points, h2_nn, "b-", {{"label", "NN on simulations"}, {"linewidth", "2"}});
    plt::plot(time_points, h2_seasonal, "g--", {{"label", "NN with seasonality"}, {"linewidth", "2"}});
    plt::plot(time_points, h2_multi, "r-.", {{"label", "Multi-preference NN"}, {"linewidth", "2"}});
    plt::plot(time_points, h2_subdiff, "m:", {{"label", "NN with γ=3/2"}, {"linewidth", "2"}});
    plt::xlabel("Time");
    plt::ylabel("h2(t)");
    plt::legend();
    plt::grid(true);
    
    // Plot R-squared values
    plt::subplot(2, 2, 3);
    plt::plot(time_points, r_squared_closed, "k-", {{"label", "Closed-form solution"}, {"linewidth", "2"}});
    plt::plot(time_points, r_squared_nn, "b-", {{"label", "NN on simulations"}, {"linewidth", "2"}});
    plt::plot(time_points, r_squared_seasonal, "g--", {{"label", "NN with seasonality"}, {"linewidth", "2"}});
    plt::plot(time_points, r_squared_multi, "r-.", {{"label", "Multi-preference NN"}, {"linewidth", "2"}});
    plt::plot(time_points, r_squared_subdiff, "m:", {{"label", "NN with γ=3/2"}, {"linewidth", "2"}});
    plt::xlabel("Time");
    plt::ylabel("R²(t)");
    plt::legend();
    plt::grid(true);
    
    // Save the figure
    plt::save("trading_controls_comparison.png");
    
    // Print final metrics
    std::cout << "\nFinal Results:" << std::endl;
    
    // Simulate trajectories
    double inventory = -100.0;
    
    // Define control functions
    auto cf_control = [&closed_form](const std::vector<double>& state) {
        return closed_form->control(state);
    };
    
    auto nn_control = [&neural_net](const std::vector<double>& state) {
        return neural_net->control(state);
    };
    
    auto multi_control = [&multi_pref_net](const std::vector<double>& state) {
        return multi_pref_net->control(state);
    };
    
    auto seasonal_control = [&neural_net_seasonal](const std::vector<double>& state) {
        return neural_net_seasonal->control(state);
    };
    
    auto subdiff_control = [&neural_net_subdiff](const std::vector<double>& state) {
        return neural_net_subdiff->control(state);
    };
    
    // Simulate trajectories
    auto cf_traj = simulator->simulateTrajectory(cf_control, inventory);
    auto nn_traj = simulator->simulateTrajectory(nn_control, inventory);
    auto multi_traj = simulator->simulateTrajectory(multi_control, inventory);
    auto seasonal_traj = simulator_seasonal->simulateTrajectory(seasonal_control, inventory);
    auto subdiff_traj = simulator_subdiff->simulateTrajectory(subdiff_control, inventory);
    
    // Print final metrics
    std::cout << "Closed-form final inventory: " << cf_traj.inventory_path.back() << std::endl;
    std::cout << "Neural Net final inventory: " << nn_traj.inventory_path.back() << std::endl;
    std::cout << "Multi-pref Net final inventory: " << multi_traj.inventory_path.back() << std::endl;
    std::cout << "Seasonal Net final inventory: " << seasonal_traj.inventory_path.back() << std::endl;
    std::cout << "Sub-diffusive Net final inventory: " << subdiff_traj.inventory_path.back() << std::endl;
    
    std::cout << "\nClosed-form total reward: " << cf_traj.total_reward << std::endl;
    std::cout << "Neural Net total reward: " << nn_traj.total_reward << std::endl;
    std::cout << "Multi-pref Net total reward: " << multi_traj.total_reward << std::endl;
    std::cout << "Seasonal Net total reward: " << seasonal_traj.total_reward << std::endl;
    std::cout << "Sub-diffusive Net total reward: " << subdiff_traj.total_reward << std::endl;
    
    return 0;
}
