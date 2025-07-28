#include <iostream>
#include <cmath>

int main() {
    std::cout << "Testing basic calculations..." << std::endl;
    
    double initial_capital = 100000.0;
    double cash = 95000.0;
    double position = 950.0;
    double current_price = 102.0;
    
    std::cout << "Initial capital: " << initial_capital << std::endl;
    std::cout << "Cash: " << cash << std::endl;
    std::cout << "Position: " << position << std::endl;
    std::cout << "Current price: " << current_price << std::endl;
    
    double current_value = cash + position * current_price;
    std::cout << "Current value: " << current_value << std::endl;
    
    double pnl = current_value - initial_capital;
    std::cout << "PnL: " << pnl << std::endl;
    
    double return_pct = (current_value - initial_capital) / initial_capital * 100.0;
    std::cout << "Return %: " << return_pct << std::endl;
    
    // Test with problematic values
    std::cout << "\nTesting edge cases:" << std::endl;
    
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    double inf_val = std::numeric_limits<double>::infinity();
    
    std::cout << "NaN: " << nan_val << std::endl;
    std::cout << "Inf: " << inf_val << std::endl;
    std::cout << "Is NaN finite? " << std::isfinite(nan_val) << std::endl;
    std::cout << "Is Inf finite? " << std::isfinite(inf_val) << std::endl;
    
    return 0;
}
