#include "../include/OptimalExecutionSimulator.h"
#include <cmath>
#include <random>
#include <chrono>

OptimalExecutionSimulator::OptimalExecutionSimulator(
    int T, double dt, double initial_price, double vol, double alpha, 
    double kappa, double A, double phi, double gamma, bool with_seasonality)
    : T(T), dt(dt), initial_price(initial_price), vol(vol), alpha(alpha),
      kappa(kappa), A(A), phi(phi), gamma(gamma), with_seasonality(with_seasonality),
      rng(std::chrono::system_clock::now().time_since_epoch().count()),
      normal_dist(0.0, 1.0) {
    
    // Create seasonality patterns if needed
    if (with_seasonality) {
        // U-shaped volume profile
        volume_profile.resize(T);
        spread_profile.resize(T);
        
        for (int i = 0; i < T; ++i) {
            // U-shaped volume profile
            volume_profile[i] = 1.0 + 0.5 * (std::exp(-0.5 * std::pow((i - 0) / 15.0, 2)) + 
                                         0.8 * std::exp(-0.5 * std::pow((i - (T-1)) / 15.0, 2)));
            
            // Inverted U-shaped spread profile
            spread_profile[i] = 1.0 + 0.5 * (std::exp(-0.5 * std::pow((i - 0) / 15.0, 2)) + 
                                        0.8 * std::exp(-0.5 * std::pow((i - (T-1)) / 15.0, 2)));
        }
        
        // Normalize profiles
        double vol_mean = 0.0, spread_mean = 0.0;
        for (int i = 0; i < T; ++i) {
            vol_mean += volume_profile[i];
            spread_mean += spread_profile[i];
        }
        vol_mean /= T;
        spread_mean /= T;
        
        for (int i = 0; i < T; ++i) {
            volume_profile[i] /= vol_mean;
            spread_profile[i] /= spread_mean;
        }
    }
}

std::vector<double> OptimalExecutionSimulator::reset(double initial_inventory) {
    t = 0;
    inventory = initial_inventory;
    price = initial_price;
    wealth = 0.0;
    
    price_path.clear();
    inventory_path.clear();
    wealth_path.clear();
    time_path.clear();
    
    price_path.push_back(price);
    inventory_path.push_back(inventory);
    wealth_path.push_back(wealth);
    time_path.push_back(t);
    
    return getState();
}

std::vector<double> OptimalExecutionSimulator::getState() const {
    return {static_cast<double>(T - t), inventory};
}

std::tuple<std::vector<double>, double, bool, std::vector<double>> 
OptimalExecutionSimulator::step(double trading_rate) {
    // Apply seasonality if needed
    double curr_vol, curr_alpha, curr_kappa;
    if (with_seasonality) {
        double vol_factor = volume_profile[t];
        double spread_factor = spread_profile[t];
        curr_vol = vol * vol_factor;
        curr_alpha = alpha * spread_factor / vol_factor;
        curr_kappa = kappa * spread_factor / vol_factor;
    } else {
        curr_vol = vol;
        curr_alpha = alpha;
        curr_kappa = kappa;
    }
    
    // Increment time
    t += 1;
    
    // Generate price noise
    double price_noise = normal_dist(rng) * curr_vol * std::sqrt(dt);
    
    // Update price, inventory, and wealth
    price += curr_alpha * trading_rate * dt + price_noise;
    inventory += trading_rate * dt;
    wealth -= trading_rate * (price + curr_kappa * trading_rate) * dt;
    
    // Store path values
    price_path.push_back(price);
    inventory_path.push_back(inventory);
    wealth_path.push_back(wealth);
    time_path.push_back(t);
    
    // Check if the episode is done
    bool done = (t >= T);
    
    // Compute reward (negative cost for this step)
    double running_penalty = -phi * std::pow(std::abs(inventory), gamma) * dt;
    
    // Add terminal penalty if this is the final step
    double terminal_penalty = 0.0;
    if (done) {
        terminal_penalty = -A * std::pow(std::abs(inventory), gamma);
    }
    
    double reward = running_penalty + terminal_penalty;
    
    return {getState(), reward, done, {}};
}

OptimalExecutionSimulator::TrajectoryResult 
OptimalExecutionSimulator::simulateTrajectory(
    const std::function<double(const std::vector<double>&)>& control_func,
    double initial_inventory) {
    
    auto state = reset(initial_inventory);
    bool done = false;
    double total_reward = 0.0;
    std::vector<double> trading_rates;
    
    while (!done) {
        double trading_rate = control_func(state);
        trading_rates.push_back(trading_rate);
        
        auto [next_state, reward, is_done, _] = step(trading_rate);
        state = next_state;
        total_reward += reward;
        done = is_done;
    }
    
    // Add the final value of the position to the reward
    double final_inventory_value = inventory_path.back() * price_path.back();
    double final_reward = total_reward + final_inventory_value;
    
    return {
        price_path,
        inventory_path,
        wealth_path,
        time_path,
        trading_rates,
        total_reward,
        final_reward
    };
}
