#include <iostream>
#include <fstream>
#include <string>
#include <set>

int main() {
    std::cout << "Testing file reading..." << std::endl;
    
    std::set<std::string> symbols;
    std::ifstream file("excel_stocks.txt");
    
    if (!file.is_open()) {
        std::cout << "Cannot open excel_stocks.txt" << std::endl;
        return 1;
    }
    
    std::string symbol;
    int count = 0;
    while (std::getline(file, symbol)) {
        if (!symbol.empty()) {
            symbols.insert(symbol);
            count++;
            if (count <= 5) {
                std::cout << "Read: " << symbol << std::endl;
            }
        }
    }
    file.close();
    
    std::cout << "Total symbols read: " << symbols.size() << std::endl;
    std::cout << "Lines processed: " << count << std::endl;
    
    return 0;
}
