#include <iostream>
#include <fstream>
#include <string>
#include <vector>

int main() {
    std::cout << "=== Testing Excel File Reading ===" << std::endl;
    
    // Check if excel_stocks.txt exists
    std::ifstream file("excel_stocks.txt");
    if (!file.is_open()) {
        std::cout << "ERROR: excel_stocks.txt not found!" << std::endl;
        std::cout << "Please run: python read_excel_stocks.py first" << std::endl;
        return 1;
    }
    
    // Read and display contents
    std::vector<std::string> symbols;
    std::string line;
    
    while (std::getline(file, line)) {
        if (!line.empty()) {
            symbols.push_back(line);
        }
    }
    file.close();
    
    std::cout << "SUCCESS: Found excel_stocks.txt" << std::endl;
    std::cout << "Number of symbols: " << symbols.size() << std::endl;
    
    if (symbols.empty()) {
        std::cout << "WARNING: File is empty!" << std::endl;
        return 1;
    }
    
    std::cout << "\nFirst 10 symbols:" << std::endl;
    for (size_t i = 0; i < std::min(symbols.size(), size_t(10)); ++i) {
        std::cout << "  " << (i+1) << ". " << symbols[i] << std::endl;
    }
    
    if (symbols.size() > 10) {
        std::cout << "  ... and " << (symbols.size() - 10) << " more" << std::endl;
    }
    
    // Count by exchange
    int sh_count = 0, sz_count = 0;
    for (const auto& symbol : symbols) {
        if (symbol.substr(0, 2) == "sh") sh_count++;
        else if (symbol.substr(0, 2) == "sz") sz_count++;
    }
    
    std::cout << "\nExchange breakdown:" << std::endl;
    std::cout << "  Shanghai (sh): " << sh_count << std::endl;
    std::cout << "  Shenzhen (sz): " << sz_count << std::endl;
    std::cout << "  Total: " << symbols.size() << std::endl;
    
    std::cout << "\nFile is ready for mass testing!" << std::endl;
    
    return 0;
}
