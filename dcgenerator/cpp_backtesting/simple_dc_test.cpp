#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

// Simple DC Generator
class DCGenerator {
public:
    enum class DCEvent { NONE, END_UPTURN, END_DOWNTURN };
    
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
                is_upturn_ = false;
                lowest_price_ = price;
                return DCEvent::END_UPTURN;
            } else if (price > highest_price_) {
                highest_price_ = price;
            }
        } else {
            if (price >= lowest_price_ * (1.0 + threshold_)) {
                is_upturn_ = true;
                highest_price_ = price;
                return DCEvent::END_DOWNTURN;
            } else if (price < lowest_price_) {
                lowest_price_ = price;
            }
        }
        return DCEvent::NONE;
    }

private:
    double threshold_;
    bool initialized_;
    double highest_price_;
    double lowest_price_;
    bool is_upturn_;
};

int main() {
    std::cout << "=== Simple DC Generator Test ===" << std::endl;
    
    // Create simple test data
    std::vector<double> prices = {
        100.0, 100.1, 100.2, 100.3, 100.4, 100.5,  // Uptrend
        100.4, 100.3, 100.2, 100.1, 100.0, 99.9,   // Downtrend
        99.8, 99.7, 99.8, 99.9, 100.0, 100.1,      // Recovery
        100.2, 100.3, 100.4, 100.3, 100.2, 100.1   // Small moves
    };
    
    std::cout << "Test data (" << prices.size() << " points):" << std::endl;
    for (size_t i = 0; i < prices.size(); ++i) {
        std::cout << std::fixed << std::setprecision(1) << prices[i];
        if (i < prices.size() - 1) std::cout << ", ";
        if ((i + 1) % 6 == 0) std::cout << std::endl;
    }
    std::cout << std::endl;
    
    // Test with 0.5% threshold
    double threshold = 0.005; // 0.5%
    DCGenerator dc_gen(threshold);
    
    std::cout << "\nDC Events (threshold: " << threshold * 100 << "%):" << std::endl;
    
    int event_count = 0;
    for (size_t i = 0; i < prices.size(); ++i) {
        DCGenerator::DCEvent event = dc_gen.processPrice(prices[i]);
        
        if (event != DCGenerator::DCEvent::NONE) {
            event_count++;
            std::string event_name = (event == DCGenerator::DCEvent::END_UPTURN) ? "END_UPTURN" : "END_DOWNTURN";
            std::cout << "Tick " << i << ": " << event_name << " at price " << prices[i] << std::endl;
        }
    }
    
    std::cout << "\nTotal DC events: " << event_count << std::endl;
    std::cout << "Events per tick: " << (double)event_count / prices.size() << std::endl;
    
    return 0;
}
