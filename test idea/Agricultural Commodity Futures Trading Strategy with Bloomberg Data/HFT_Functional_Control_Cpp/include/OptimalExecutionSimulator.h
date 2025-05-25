#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <functional>
#include <memory>

/**
 * @class OptimalExecutionSimulator
 * @brief Simulator for the optimal execution problem with price impact
 */
class OptimalExecutionSimulator {
public:
    /**
     * @brief Constructor
     * @param T Number of time steps
     * @param dt Time step size
     * @param initial_price Initial price of the asset
     * @param vol Volatility of the asset price
     * @param alpha Permanent price impact coefficient
     * @param kappa Temporary price impact coefficient
     * @param A Terminal inventory penalty
     * @param phi Running inventory penalty
     * @param gamma Exponent for inventory penalty (2.0 for quadratic, 1.5 for sub-diffusive)
     * @param with_seasonality Whether to include intraday seasonality in the simulation
     */
    OptimalExecutionSimulator(int T = 77, double dt = 1.0, double initial_price = 100.0,
                             double vol = 0.1, double alpha = 0.1, double kappa = 0.1,
                             double A = 0.01, double phi = 0.007, double gamma = 2.0,
                             bool with_seasonality = false);

    /**
     * @brief Reset the simulation with a given initial inventory
     * @param initial_inventory Initial inventory
     * @return Current state of the simulation
     */
    std::vector<double> reset(double initial_inventory = -100.0);

    /**
     * @brief Take a step in the simulation given a trading rate
     * @param trading_rate Trading rate (speed) for this step
     * @return Tuple of (next_state, reward, done, info)
     */
    std::tuple<std::vector<double>, double, bool, std::vector<double>> step(double trading_rate);

    /**
     * @brief Simulate a full trajectory using a control function
     * @param control_func Function that maps state to trading rate
     * @param initial_inventory Initial inventory
     * @return Dictionary containing simulation results
     */
    struct TrajectoryResult {
        std::vector<double> price_path;
        std::vector<double> inventory_path;
        std::vector<double> wealth_path;
        std::vector<double> time_path;
        std::vector<double> trading_rates;
        double total_reward;
        double final_reward;
    };

    TrajectoryResult simulateTrajectory(
        const std::function<double(const std::vector<double>&)>& control_func,
        double initial_inventory = -100.0);

    // Getters for parameters
    int getT() const { return T; }
    double getDt() const { return dt; }
    double getAlpha() const { return alpha; }
    double getKappa() const { return kappa; }
    double getA() const { return A; }
    double getPhi() const { return phi; }
    double getGamma() const { return gamma; }

    // Setters for parameters
    void setA(double value) { A = value; }
    void setPhi(double value) { phi = value; }

private:
    /**
     * @brief Get the current state of the simulation
     * @return Current state vector
     */
    std::vector<double> getState() const;

    // Simulation parameters
    int T;
    double dt;
    double initial_price;
    double vol;
    double alpha;
    double kappa;
    double A;
    double phi;
    double gamma;
    bool with_seasonality;

    // Current state
    int t;
    double inventory;
    double price;
    double wealth;

    // Path storage
    std::vector<double> price_path;
    std::vector<double> inventory_path;
    std::vector<double> wealth_path;
    std::vector<double> time_path;

    // Seasonality profiles
    std::vector<double> volume_profile;
    std::vector<double> spread_profile;

    // Random number generator
    std::mt19937 rng;
    std::normal_distribution<double> normal_dist;
};
