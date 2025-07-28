#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>

// Simple DC Generator implementation
class DCGenerator {
public:
    enum class DCEvent {
        NONE,
        END_UPTURN,
        END_DOWNTURN
    };
    
    explicit DCGenerator(double threshold) : threshold_(threshold), initialized_(false) {}
    
    DCEvent processPrice(double price) {
        if (!initialized_) {
            highest_price_ = price;
            lowest_price_ = price;
            is_upturn_ = true;
            initialized_ = true;
            return DCEvent::NONE;
        }
        
        if (is_upturn_) {
            if (price <= highest_price_ * (1.0 - threshold_)) {
                // End of upturn
                is_upturn_ = false;
                lowest_price_ = price;
                return DCEvent::END_UPTURN;
            } else if (price > highest_price_) {
                highest_price_ = price;
            }
        } else {
            if (price >= lowest_price_ * (1.0 + threshold_)) {
                // End of downturn
                is_upturn_ = true;
                highest_price_ = price;
                return DCEvent::END_DOWNTURN;
            } else if (price < lowest_price_) {
                lowest_price_ = price;
            }
        }
        
        return DCEvent::NONE;
    }
    
    void reset() {
        initialized_ = false;
        highest_price_ = 0;
        lowest_price_ = 0;
        is_upturn_ = true;
    }
    
    double getThreshold() const { return threshold_; }

private:
    double threshold_;
    bool initialized_;
    double highest_price_;
    double lowest_price_;
    bool is_upturn_;
};

// Simple trading strategy
class SimpleTradingStrategy {
public:
    explicit SimpleTradingStrategy(double initial_capital) 
        : initial_capital_(initial_capital), cash_(initial_capital), position_(0), has_position_(false) {}
    
    void onDCEvent(DCGenerator::DCEvent event, double price) {
        // Safety check for invalid prices
        if (!std::isfinite(price) || price <= 0) {
            std::cout << "Warning: Invalid price " << price << ", skipping trade" << std::endl;
            return;
        }

        switch (event) {
            case DCGenerator::DCEvent::END_DOWNTURN:
                if (!has_position_ && cash_ > 0) {
                    // Buy signal
                    double cash_to_use = cash_ * 0.95; // Use 95% of available cash
                    position_ = cash_to_use / price;
                    cash_ -= cash_to_use;
                    has_position_ = true;

                    trades_.push_back({price, position_, "BUY"});
                    std::cout << "BUY at " << std::fixed << std::setprecision(2)
                              << price << ", Position: " << std::setprecision(4) << position_ << std::endl;
                }
                break;

            case DCGenerator::DCEvent::END_UPTURN:
                if (has_position_ && position_ > 0) {
                    // Sell signal
                    cash_ += position_ * price;

                    trades_.push_back({price, position_, "SELL"});
                    std::cout << "SELL at " << std::fixed << std::setprecision(2)
                              << price << ", Cash: " << std::setprecision(2) << cash_ << std::endl;

                    position_ = 0;
                    has_position_ = false;
                }
                break;

            default:
                break;
        }
    }
    
    double getCurrentValue(double current_price) const {
        return cash_ + position_ * current_price;
    }
    
    double getTotalReturn(double current_price) const {
        double current_value = getCurrentValue(current_price);
        return (current_value - initial_capital_) / initial_capital_ * 100.0;
    }
    
    int getTradeCount() const { return trades_.size(); }
    
    void printSummary(double final_price) const {
        double final_value = getCurrentValue(final_price);
        double total_return = getTotalReturn(final_price);
        
        std::cout << "\n=== Trading Summary ===" << std::endl;
        std::cout << "Initial Capital: $" << std::fixed << std::setprecision(2) << initial_capital_ << std::endl;
        std::cout << "Final Value: $" << final_value << std::endl;
        std::cout << "Total Return: " << total_return << "%" << std::endl;
        std::cout << "Number of Trades: " << trades_.size() << std::endl;
        
        if (!trades_.empty()) {
            std::cout << "First Trade: " << trades_[0].action << " at " << trades_[0].price << std::endl;
            std::cout << "Last Trade: " << trades_.back().action << " at " << trades_.back().price << std::endl;
        }
    }

private:
    struct Trade {
        double price;
        double quantity;
        std::string action;
    };
    
    double initial_capital_;
    double cash_;
    double position_;
    bool has_position_;
    std::vector<Trade> trades_;
};

// Generate sample high-frequency price data
std::vector<double> generateSamplePriceData(int num_points, double initial_price = 100.0) {
    std::vector<double> prices;
    prices.reserve(num_points);

    // Use a simple deterministic approach to avoid random number issues
    double current_price = initial_price;
    prices.push_back(current_price);

    // Simple controlled price movements to avoid overflow
    for (int i = 1; i < num_points; ++i) {
        // Create small, bounded movements
        double base_change = 0.0001 * ((i * 17) % 21 - 10); // Range: -0.001 to +0.001
        double wave = 0.0002 * std::sin(i * 0.001); // Small oscillations
        double noise = 0.0001 * (i % 7 - 3); // Simple noise pattern

        double total_change = base_change + wave + noise;

        // Clamp the change to prevent extreme movements
        total_change = std::max(-0.005, std::min(0.005, total_change)); // Max 0.5% change per tick

        current_price = current_price * (1.0 + total_change);

        // Ensure price stays within reasonable bounds
        if (current_price <= 0 || current_price > 10000.0 || !std::isfinite(current_price)) {
            current_price = initial_price; // Reset to initial if something goes wrong
        }

        prices.push_back(current_price);
    }

    return prices;
}

void testDCAlgorithm(const std::vector<double>& prices, double threshold) {
    std::cout << "\n=== Testing DC Algorithm (Threshold: " << threshold * 100 << "%) ===" << std::endl;
    
    DCGenerator dc_gen(threshold);
    SimpleTradingStrategy strategy(100000.0);
    
    int dc_events = 0;
    int upturn_ends = 0;
    int downturn_ends = 0;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < prices.size(); ++i) {
        DCGenerator::DCEvent event = dc_gen.processPrice(prices[i]);
        
        if (event != DCGenerator::DCEvent::NONE) {
            dc_events++;
            if (event == DCGenerator::DCEvent::END_UPTURN) {
                upturn_ends++;
            } else if (event == DCGenerator::DCEvent::END_DOWNTURN) {
                downturn_ends++;
            }
            
            strategy.onDCEvent(event, prices[i]);
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "Processing time: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Total DC events: " << dc_events << std::endl;
    std::cout << "End upturn events: " << upturn_ends << std::endl;
    std::cout << "End downturn events: " << downturn_ends << std::endl;
    std::cout << "Events per 1000 ticks: " << (dc_events * 1000.0 / prices.size()) << std::endl;
    
    strategy.printSummary(prices.back());
}

void compareThresholds(const std::vector<double>& prices) {
    std::vector<double> thresholds = {0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01};
    
    std::cout << "\n=== Threshold Comparison ===" << std::endl;
    std::cout << std::setw(12) << "Threshold" 
              << std::setw(12) << "Events" 
              << std::setw(12) << "Return %" 
              << std::setw(12) << "Trades" << std::endl;
    std::cout << std::string(48, '-') << std::endl;
    
    for (double threshold : thresholds) {
        DCGenerator dc_gen(threshold);
        SimpleTradingStrategy strategy(100000.0);
        
        int dc_events = 0;
        
        for (double price : prices) {
            DCGenerator::DCEvent event = dc_gen.processPrice(price);
            if (event != DCGenerator::DCEvent::NONE) {
                dc_events++;
                strategy.onDCEvent(event, price);
            }
        }
        
        double return_pct = strategy.getTotalReturn(prices.back());
        
        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(12) << threshold * 100
                  << std::setw(12) << dc_events
                  << std::setw(12) << std::setprecision(2) << return_pct
                  << std::setw(12) << strategy.getTradeCount() << std::endl;
    }
}

int main() {
    std::cout << "=== DC Generator High-Frequency Trading Demo ===" << std::endl;
    
    // Generate sample high-frequency data
    const int num_ticks = 100000; // 100k ticks to simulate HFT data
    std::cout << "Generating " << num_ticks << " sample price ticks..." << std::endl;
    
    auto prices = generateSamplePriceData(num_ticks, 100.0);
    
    std::cout << "Price range: " << std::fixed << std::setprecision(2) 
              << *std::min_element(prices.begin(), prices.end()) << " to " 
              << *std::max_element(prices.begin(), prices.end()) << std::endl;
    
    // Test with different thresholds
    testDCAlgorithm(prices, 0.001); // 0.1%
    testDCAlgorithm(prices, 0.002); // 0.2%
    testDCAlgorithm(prices, 0.005); // 0.5%
    
    // Compare all thresholds
    compareThresholds(prices);
    
    std::cout << "\n=== Performance Notes ===" << std::endl;
    std::cout << "- This demo processes " << num_ticks << " ticks in microseconds" << std::endl;
    std::cout << "- Real HFT systems can process millions of ticks per second" << std::endl;
    std::cout << "- The DC algorithm has O(1) complexity per tick" << std::endl;
    std::cout << "- Memory usage is constant regardless of data size" << std::endl;
    
    return 0;
}
