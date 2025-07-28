#include <iostream>
#include <vector>
#include <string>
#include <windows.h>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

// SQLite function pointers (same as before)
typedef struct sqlite3 sqlite3;
typedef struct sqlite3_stmt sqlite3_stmt;
typedef int (*sqlite3_open_func)(const char*, sqlite3**);
typedef int (*sqlite3_close_func)(sqlite3*);
typedef int (*sqlite3_prepare_v2_func)(sqlite3*, const char*, int, sqlite3_stmt**, const char**);
typedef int (*sqlite3_step_func)(sqlite3_stmt*);
typedef int (*sqlite3_finalize_func)(sqlite3_stmt*);
typedef const unsigned char* (*sqlite3_column_text_func)(sqlite3_stmt*, int);
typedef double (*sqlite3_column_double_func)(sqlite3_stmt*, int);
typedef int (*sqlite3_bind_text_func)(sqlite3_stmt*, int, const char*, int, void(*)(void*));
typedef const char* (*sqlite3_errmsg_func)(sqlite3*);

#define SQLITE_OK           0
#define SQLITE_ROW          100

// Global function pointers
sqlite3_open_func sqlite3_open_ptr = nullptr;
sqlite3_close_func sqlite3_close_ptr = nullptr;
sqlite3_prepare_v2_func sqlite3_prepare_v2_ptr = nullptr;
sqlite3_step_func sqlite3_step_ptr = nullptr;
sqlite3_finalize_func sqlite3_finalize_ptr = nullptr;
sqlite3_column_text_func sqlite3_column_text_ptr = nullptr;
sqlite3_column_double_func sqlite3_column_double_ptr = nullptr;
sqlite3_bind_text_func sqlite3_bind_text_ptr = nullptr;
sqlite3_errmsg_func sqlite3_errmsg_ptr = nullptr;

bool loadSQLite() {
    HMODULE hModule = LoadLibraryA("C:/sqlite3/sqlite3.dll");
    if (!hModule) {
        hModule = LoadLibraryA("sqlite3.dll");
    }
    if (!hModule) {
        std::cout << "Could not load SQLite library" << std::endl;
        return false;
    }
    
    sqlite3_open_ptr = (sqlite3_open_func)GetProcAddress(hModule, "sqlite3_open");
    sqlite3_close_ptr = (sqlite3_close_func)GetProcAddress(hModule, "sqlite3_close");
    sqlite3_prepare_v2_ptr = (sqlite3_prepare_v2_func)GetProcAddress(hModule, "sqlite3_prepare_v2");
    sqlite3_step_ptr = (sqlite3_step_func)GetProcAddress(hModule, "sqlite3_step");
    sqlite3_finalize_ptr = (sqlite3_finalize_func)GetProcAddress(hModule, "sqlite3_finalize");
    sqlite3_column_text_ptr = (sqlite3_column_text_func)GetProcAddress(hModule, "sqlite3_column_text");
    sqlite3_column_double_ptr = (sqlite3_column_double_func)GetProcAddress(hModule, "sqlite3_column_double");
    sqlite3_bind_text_ptr = (sqlite3_bind_text_func)GetProcAddress(hModule, "sqlite3_bind_text");
    sqlite3_errmsg_ptr = (sqlite3_errmsg_func)GetProcAddress(hModule, "sqlite3_errmsg");
    
    return (sqlite3_open_ptr && sqlite3_close_ptr && sqlite3_prepare_v2_ptr && 
            sqlite3_step_ptr && sqlite3_finalize_ptr);
}

// Correct DCGenerator from China market system (same as before)
class DCGenerator {
public:
    enum class DCEvent {
        NONE,
        END_UPTURN,
        END_DOWNTURN
    };
    
    DCGenerator(double threshold) : threshold_(threshold), initialized_(false) {
        if (threshold <= 0.0 || threshold >= 1.0) {
            throw std::invalid_argument("DC threshold must be between 0 and 1");
        }
        reset();
    }
    
    DCEvent processPrice(double price) {
        if (price <= 0.0) {
            return DCEvent::NONE;
        }
        
        if (!initialized_) {
            highest_price_ = price;
            lowest_price_ = price;
            is_upturn_ = true;
            initialized_ = true;
            return DCEvent::NONE;
        }
        
        DCEvent event = checkForEvents(price);
        updateState(price, event);
        
        return event;
    }
    
    void reset() {
        highest_price_ = 0.0;
        lowest_price_ = 0.0;
        is_upturn_ = true;
        initialized_ = false;
    }

private:
    double threshold_;
    double highest_price_;
    double lowest_price_;
    bool is_upturn_;
    bool initialized_;
    
    DCEvent checkForEvents(double current_price) {
        if (is_upturn_) {
            if (current_price <= highest_price_ * (1.0 - threshold_)) {
                return DCEvent::END_UPTURN;
            } else if (current_price > highest_price_) {
                return DCEvent::NONE;
            }
        } else {
            if (current_price >= lowest_price_ * (1.0 + threshold_)) {
                return DCEvent::END_DOWNTURN;
            } else if (current_price < lowest_price_) {
                return DCEvent::NONE;
            }
        }
        
        return DCEvent::NONE;
    }
    
    void updateState(double current_price, DCEvent event) {
        switch (event) {
            case DCEvent::END_UPTURN:
                is_upturn_ = false;
                lowest_price_ = current_price;
                break;
                
            case DCEvent::END_DOWNTURN:
                is_upturn_ = true;
                highest_price_ = current_price;
                break;
                
            default:
                if (is_upturn_ && current_price > highest_price_) {
                    highest_price_ = current_price;
                } else if (!is_upturn_ && current_price < lowest_price_) {
                    lowest_price_ = current_price;
                }
                break;
        }
    }
};

// Simple DC Strategy
class SimpleDCStrategy {
private:
    double initial_capital_;
    double current_cash_;
    double shares_held_;
    int trade_count_;
    
public:
    SimpleDCStrategy(double capital) : initial_capital_(capital), current_cash_(capital), shares_held_(0), trade_count_(0) {}
    
    void onDCEvent(DCGenerator::DCEvent event, double price) {
        if (event == DCGenerator::DCEvent::END_DOWNTURN) {
            if (current_cash_ > 0 && price > 0) {
                shares_held_ = current_cash_ / price;
                current_cash_ = 0;
                trade_count_++;
            }
        } else if (event == DCGenerator::DCEvent::END_UPTURN) {
            if (shares_held_ > 0 && price > 0) {
                current_cash_ = shares_held_ * price;
                shares_held_ = 0;
                trade_count_++;
            }
        }
    }
    
    double getCurrentValue(double current_price) const {
        return current_cash_ + (shares_held_ * current_price);
    }
    
    double getPnL(double current_price) const {
        return getCurrentValue(current_price) - initial_capital_;
    }
    
    double getTotalReturn(double current_price) const {
        return (getCurrentValue(current_price) - initial_capital_) / initial_capital_ * 100.0;
    }
    
    int getTradeCount() const { return trade_count_; }
};

// Get all unique symbols from databases
std::vector<std::string> getAllChinaSymbols() {
    std::vector<std::string> symbols;
    
    std::vector<std::string> db_paths = {
        "I:\\zhubi\\cpp_implementation\\sqlite_databases\\2018\\2018_01.db",
        "I:\\zhubi\\cpp_implementation\\sqlite_databases\\2018\\2018_02.db"
    };
    
    std::cout << "Scanning databases for all available symbols..." << std::endl;
    
    for (const auto& db_path : db_paths) {
        sqlite3* db;
        int rc = sqlite3_open_ptr(db_path.c_str(), &db);
        
        if (rc != SQLITE_OK) {
            continue;
        }
        
        std::string sql = "SELECT DISTINCT symbol FROM trade ORDER BY symbol;";
        sqlite3_stmt* stmt;
        
        rc = sqlite3_prepare_v2_ptr(db, sql.c_str(), -1, &stmt, nullptr);
        if (rc == SQLITE_OK) {
            while ((rc = sqlite3_step_ptr(stmt)) == SQLITE_ROW) {
                const char* symbol = reinterpret_cast<const char*>(sqlite3_column_text_ptr(stmt, 0));
                if (symbol) {
                    std::string sym_str(symbol);
                    if (std::find(symbols.begin(), symbols.end(), sym_str) == symbols.end()) {
                        symbols.push_back(sym_str);
                    }
                }
            }
            sqlite3_finalize_ptr(stmt);
        }
        
        sqlite3_close_ptr(db);
        break; // Just get symbols from first database
    }
    
    std::cout << "Found " << symbols.size() << " unique symbols" << std::endl;
    return symbols;
}

// Load data for a specific symbol (simplified version)
std::vector<double> loadSymbolData(const std::string& symbol) {
    std::vector<double> prices;
    
    std::vector<std::string> db_paths = {
        "I:\\zhubi\\cpp_implementation\\sqlite_databases\\2018\\2018_01.db",
        "I:\\zhubi\\cpp_implementation\\sqlite_databases\\2018\\2018_02.db"
    };
    
    for (const auto& db_path : db_paths) {
        sqlite3* db;
        int rc = sqlite3_open_ptr(db_path.c_str(), &db);
        
        if (rc != SQLITE_OK) {
            continue;
        }
        
        std::string sql = "SELECT price FROM trade WHERE symbol = ? ORDER BY date, time LIMIT 20000;";
        sqlite3_stmt* stmt;
        
        rc = sqlite3_prepare_v2_ptr(db, sql.c_str(), -1, &stmt, nullptr);
        if (rc == SQLITE_OK) {
            sqlite3_bind_text_ptr(stmt, 1, symbol.c_str(), -1, nullptr);
            
            while ((rc = sqlite3_step_ptr(stmt)) == SQLITE_ROW) {
                double price = sqlite3_column_double_ptr(stmt, 0);
                if (price > 0 && std::isfinite(price)) {
                    prices.push_back(price);
                }
            }
            sqlite3_finalize_ptr(stmt);
        }
        
        sqlite3_close_ptr(db);
        
        if (prices.size() >= 10000) break; // Enough data
    }
    
    return prices;
}

// Test a single symbol
struct TestResult {
    std::string symbol;
    int data_points;
    double min_price;
    double max_price;
    double price_range_pct;
    int trades_05;
    double return_05;
    int trades_10;
    double return_10;
    bool valid;
};

TestResult testSymbol(const std::string& symbol) {
    TestResult result;
    result.symbol = symbol;
    result.valid = false;
    
    auto prices = loadSymbolData(symbol);
    
    if (prices.size() < 1000) {
        return result; // Not enough data
    }
    
    result.data_points = prices.size();
    result.min_price = *std::min_element(prices.begin(), prices.end());
    result.max_price = *std::max_element(prices.begin(), prices.end());
    result.price_range_pct = ((result.max_price - result.min_price) / result.min_price) * 100.0;
    
    // Test 0.5% threshold
    DCGenerator dc_gen_05(0.005);
    SimpleDCStrategy strategy_05(100000.0);
    
    for (double price : prices) {
        DCGenerator::DCEvent event = dc_gen_05.processPrice(price);
        if (event != DCGenerator::DCEvent::NONE) {
            strategy_05.onDCEvent(event, price);
        }
    }
    
    result.trades_05 = strategy_05.getTradeCount();
    result.return_05 = strategy_05.getTotalReturn(prices.back());
    
    // Test 1.0% threshold
    DCGenerator dc_gen_10(0.01);
    SimpleDCStrategy strategy_10(100000.0);
    
    for (double price : prices) {
        DCGenerator::DCEvent event = dc_gen_10.processPrice(price);
        if (event != DCGenerator::DCEvent::NONE) {
            strategy_10.onDCEvent(event, price);
        }
    }
    
    result.trades_10 = strategy_10.getTradeCount();
    result.return_10 = strategy_10.getTotalReturn(prices.back());
    
    result.valid = true;
    return result;
}

int main() {
    std::cout << "=== China A-Share Mass DC Testing ===" << std::endl;
    
    if (!loadSQLite()) {
        std::cout << "Failed to load SQLite library" << std::endl;
        return 1;
    }
    
    // Get all symbols
    auto symbols = getAllChinaSymbols();
    
    if (symbols.empty()) {
        std::cout << "No symbols found!" << std::endl;
        return 1;
    }
    
    std::cout << "\nTesting " << symbols.size() << " symbols..." << std::endl;
    std::cout << "This may take several minutes..." << std::endl;
    
    // Open output file
    std::ofstream outfile("china_mass_dc_results.csv");
    outfile << "Symbol,DataPoints,MinPrice,MaxPrice,PriceRange%,Trades_0.5%,Return_0.5%,Trades_1.0%,Return_1.0%" << std::endl;
    
    std::vector<TestResult> results;
    int processed = 0;
    int valid_results = 0;
    
    for (const auto& symbol : symbols) {
        processed++;
        
        if (processed % 10 == 0) {
            std::cout << "Processed " << processed << "/" << symbols.size() << " symbols..." << std::endl;
        }
        
        TestResult result = testSymbol(symbol);
        
        if (result.valid) {
            results.push_back(result);
            valid_results++;
            
            // Write to CSV
            outfile << result.symbol << ","
                    << result.data_points << ","
                    << std::fixed << std::setprecision(2) << result.min_price << ","
                    << result.max_price << ","
                    << std::setprecision(1) << result.price_range_pct << ","
                    << result.trades_05 << ","
                    << std::setprecision(2) << result.return_05 << ","
                    << result.trades_10 << ","
                    << result.return_10 << std::endl;
        }
        
        // Limit to first 50 symbols for testing
        if (processed >= 50) {
            std::cout << "Limiting to first 50 symbols for testing..." << std::endl;
            break;
        }
    }
    
    outfile.close();
    
    std::cout << "\n=== SUMMARY ===" << std::endl;
    std::cout << "Total symbols processed: " << processed << std::endl;
    std::cout << "Valid results: " << valid_results << std::endl;
    std::cout << "Results saved to: china_mass_dc_results.csv" << std::endl;
    
    // Show top performers
    if (!results.empty()) {
        std::sort(results.begin(), results.end(), [](const TestResult& a, const TestResult& b) {
            return a.return_05 > b.return_05;
        });
        
        std::cout << "\nTop 10 performers (0.5% threshold):" << std::endl;
        std::cout << std::setw(12) << "Symbol" 
                  << std::setw(10) << "Trades" 
                  << std::setw(12) << "Return %" 
                  << std::setw(12) << "Range %" << std::endl;
        std::cout << std::string(45, '-') << std::endl;
        
        for (size_t i = 0; i < std::min(size_t(10), results.size()); ++i) {
            const auto& r = results[i];
            std::cout << std::setw(12) << r.symbol
                      << std::setw(10) << r.trades_05
                      << std::setw(11) << std::fixed << std::setprecision(2) << r.return_05 << "%"
                      << std::setw(11) << std::setprecision(1) << r.price_range_pct << "%" << std::endl;
        }
    }
    
    return 0;
}
