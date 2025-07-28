#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <limits>

// Simple test to debug the NaN issue
class SimpleTradingStrategy {
public:
    explicit SimpleTradingStrategy(double initial_capital) 
        : initial_capital_(initial_capital), cash_(initial_capital), position_(0), trade_count_(0) {}
    
    void buySignal(double price) {
        if (position_ == 0 && cash_ > 1.0 && price > 0) {
            double cash_to_use = cash_ * 0.95;
            position_ = cash_to_use / price;
            cash_ -= cash_to_use;
            trade_count_++;
            
            std::cout << "BUY: Price=" << price << ", Cash_used=" << cash_to_use 
                      << ", Position=" << position_ << ", Remaining_cash=" << cash_ << std::endl;
        }
    }
    
    void sellSignal(double price) {
        if (position_ > 0 && price > 0) {
            double sale_value = position_ * price;
            cash_ += sale_value;
            position_ = 0;
            trade_count_++;
            
            std::cout << "SELL: Price=" << price << ", Sale_value=" << sale_value 
                      << ", Total_cash=" << cash_ << std::endl;
        }
    }
    
    double getCurrentValue(double current_price) const {
        return cash_ + position_ * current_price;
    }
    
    double getPnL(double current_price) const {
        return getCurrentValue(current_price) - initial_capital_;
    }
    
    double getTotalReturn(double current_price) const {
        return (getCurrentValue(current_price) - initial_capital_) / initial_capital_ * 100.0;
    }
    
    void printStatus(double current_price) const {
        std::cout << "Status: Cash=" << cash_ << ", Position=" << position_ 
                  << ", Current_price=" << current_price 
                  << ", Total_value=" << getCurrentValue(current_price)
                  << ", PnL=" << getPnL(current_price)
                  << ", Return=" << getTotalReturn(current_price) << "%" << std::endl;
    }

private:
    double initial_capital_;
    double cash_;
    double position_;
    int trade_count_;
};

int main() {
    std::cout << "=== Debug DC Trading Strategy ===" << std::endl;
    
    // Test with simple price data
    std::vector<double> test_prices = {100.0, 101.0, 102.0, 101.5, 100.5, 99.0, 100.5, 102.0};
    
    SimpleTradingStrategy strategy(100000.0);
    
    std::cout << "Initial status:" << std::endl;
    strategy.printStatus(test_prices[0]);
    
    // Simulate some trades
    std::cout << "\nSimulating trades:" << std::endl;
    
    // Buy at price 100
    strategy.buySignal(test_prices[0]);
    strategy.printStatus(test_prices[0]);
    
    // Check value at different prices
    for (size_t i = 1; i < test_prices.size(); ++i) {
        std::cout << "\nPrice update to " << test_prices[i] << ":" << std::endl;
        strategy.printStatus(test_prices[i]);
        
        // Sell at price 102
        if (test_prices[i] >= 102.0) {
            strategy.sellSignal(test_prices[i]);
            strategy.printStatus(test_prices[i]);
            break;
        }
    }
    
    std::cout << "\n=== Final Results ===" << std::endl;
    double final_price = test_prices.back();
    std::cout << "Final price: " << final_price << std::endl;
    std::cout << "Final PnL: $" << strategy.getPnL(final_price) << std::endl;
    std::cout << "Final Return: " << strategy.getTotalReturn(final_price) << "%" << std::endl;
    
    // Test edge cases
    std::cout << "\n=== Testing Edge Cases ===" << std::endl;
    
    // Test with zero price
    std::cout << "Testing with zero price:" << std::endl;
    strategy.printStatus(0.0);
    
    // Test with very large price
    std::cout << "Testing with large price:" << std::endl;
    strategy.printStatus(1000000.0);
    
    // Test with NaN price
    std::cout << "Testing with NaN price:" << std::endl;
    strategy.printStatus(std::numeric_limits<double>::quiet_NaN());
    
    return 0;
}
