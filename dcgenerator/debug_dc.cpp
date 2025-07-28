#include <iostream>
#include <vector>
#include <iomanip>

// Simple DC Generator for debugging
class DebugDCGenerator {
private:
    double threshold;
    double extreme_price;
    bool in_upturn;
    bool initialized;
    
public:
    enum class DCEvent {
        NONE,
        END_UPTURN,
        END_DOWNTURN
    };
    
    DebugDCGenerator(double thresh) : threshold(thresh), extreme_price(0.0), in_upturn(true), initialized(false) {
        std::cout << "DC Generator created with threshold: " << std::fixed << std::setprecision(4) << threshold << " (" << (threshold * 100) << "%)" << std::endl;
    }
    
    DCEvent processPrice(double price) {
        if (!initialized) {
            extreme_price = price;
            initialized = true;
            std::cout << "Initialized with price: $" << std::fixed << std::setprecision(2) << price << std::endl;
            return DCEvent::NONE;
        }
        
        if (in_upturn) {
            if (price > extreme_price) {
                extreme_price = price;
                return DCEvent::NONE;
            } else {
                double decline = (extreme_price - price) / extreme_price;
                if (decline >= threshold) {
                    std::cout << "DC DOWNTURN detected! Price: $" << price << ", Extreme: $" << extreme_price 
                              << ", Decline: " << (decline * 100) << "%" << std::endl;
                    in_upturn = false;
                    extreme_price = price;
                    return DCEvent::END_UPTURN;
                }
            }
        } else {
            if (price < extreme_price) {
                extreme_price = price;
                return DCEvent::NONE;
            } else {
                double rise = (price - extreme_price) / extreme_price;
                if (rise >= threshold) {
                    std::cout << "DC UPTURN detected! Price: $" << price << ", Extreme: $" << extreme_price 
                              << ", Rise: " << (rise * 100) << "%" << std::endl;
                    in_upturn = true;
                    extreme_price = price;
                    return DCEvent::END_DOWNTURN;
                }
            }
        }
        
        return DCEvent::NONE;
    }
};

int main() {
    std::cout << "=== DC Generator Debug Test ===" << std::endl;
    
    // Test with AAPL-like prices
    std::vector<double> test_prices = {
        15.00, 15.01, 15.02, 15.05, 15.10,  // Small upturn
        15.08, 15.05, 15.00, 14.95, 14.90,  // Downturn
        14.92, 14.95, 15.00, 15.05, 15.10,  // Recovery
        15.15, 15.20, 15.25, 15.30, 15.35   // Continued upturn
    };
    
    std::cout << "\nTest prices: ";
    for (double p : test_prices) {
        std::cout << "$" << std::fixed << std::setprecision(2) << p << " ";
    }
    std::cout << std::endl << std::endl;
    
    // Test with different thresholds
    std::vector<double> thresholds = {0.001, 0.005, 0.01}; // 0.1%, 0.5%, 1%
    
    for (double thresh : thresholds) {
        std::cout << "\n=== Testing threshold: " << (thresh * 100) << "% ===" << std::endl;
        DebugDCGenerator dc(thresh);
        
        int events = 0;
        for (size_t i = 0; i < test_prices.size(); ++i) {
            auto event = dc.processPrice(test_prices[i]);
            if (event != DebugDCGenerator::DCEvent::NONE) {
                events++;
                std::cout << "  Event " << events << " at tick " << i << std::endl;
            }
        }
        
        std::cout << "Total events: " << events << std::endl;
    }
    
    return 0;
}
