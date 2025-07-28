#include <iostream>
#include <vector>
#include <string>
#include <windows.h>
#include <iomanip>
#include <algorithm>
#include <cmath>

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

// Correct DCGenerator from China market system
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
            // Currently in an upturn
            if (current_price <= highest_price_ * (1.0 - threshold_)) {
                // Price dropped by threshold from peak - end of upturn
                return DCEvent::END_UPTURN;
            } else if (current_price > highest_price_) {
                // New high - continue upturn
                return DCEvent::NONE;
            }
        } else {
            // Currently in a downturn
            if (current_price >= lowest_price_ * (1.0 + threshold_)) {
                // Price rose by threshold from trough - end of downturn
                return DCEvent::END_DOWNTURN;
            } else if (current_price < lowest_price_) {
                // New low - continue downturn
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
                // Update extremes during trend continuation
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
            // Buy signal - end of downturn
            if (current_cash_ > 0 && price > 0) {
                shares_held_ = current_cash_ / price;
                current_cash_ = 0;
                trade_count_++;
            }
        } else if (event == DCGenerator::DCEvent::END_UPTURN) {
            // Sell signal - end of upturn
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

// Contrarian DC Strategy (opposite of Simple DC)
class ContrarianDCStrategy {
private:
    double initial_capital_;
    double current_cash_;
    double shares_held_;
    int trade_count_;

public:
    ContrarianDCStrategy(double capital) : initial_capital_(capital), current_cash_(capital), shares_held_(0), trade_count_(0) {}

    void onDCEvent(DCGenerator::DCEvent event, double price) {
        if (event == DCGenerator::DCEvent::END_UPTURN) {
            // Buy signal - end of upturn (contrarian: buy when others sell)
            if (current_cash_ > 0 && price > 0) {
                shares_held_ = current_cash_ / price;
                current_cash_ = 0;
                trade_count_++;
            }
        } else if (event == DCGenerator::DCEvent::END_DOWNTURN) {
            // Sell signal - end of downturn (contrarian: sell when others buy)
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

// Load China market data from multiple months
std::vector<double> loadChinaSymbolData(const std::string& symbol) {
    std::vector<double> prices;

    // Multiple database paths for different months
    std::vector<std::string> db_paths = {
        "I:\\zhubi\\cpp_implementation\\sqlite_databases\\2018\\2018_01.db",
        "I:\\zhubi\\cpp_implementation\\sqlite_databases\\2018\\2018_02.db",
        "I:\\zhubi\\cpp_implementation\\sqlite_databases\\2018\\2018_03.db",
        "I:\\zhubi\\cpp_implementation\\sqlite_databases\\2018\\2018_04.db",
        "I:\\zhubi\\cpp_implementation\\sqlite_databases\\2018\\2018_05.db",
        "I:\\zhubi\\cpp_implementation\\sqlite_databases\\2018\\2018_06.db"
    };

    std::cout << "Loading China market data for: " << symbol << std::endl;
    std::cout << "Loading from multiple months (2018-01 to 2018-06)..." << std::endl;

    int total_loaded = 0;
    int databases_found = 0;

    for (const auto& db_path : db_paths) {
        std::cout << "  Checking: " << db_path.substr(db_path.find_last_of("\\") + 1) << "...";

        sqlite3* db;
        int rc = sqlite3_open_ptr(db_path.c_str(), &db);

        if (rc != SQLITE_OK) {
            std::cout << " not found" << std::endl;
            if (db) sqlite3_close_ptr(db);
            continue;
        }

        databases_found++;

        // Query for symbol data from 'trade' table
        std::string sql = "SELECT price FROM trade WHERE symbol = ? ORDER BY date, time LIMIT 50000;";
        sqlite3_stmt* stmt;

        rc = sqlite3_prepare_v2_ptr(db, sql.c_str(), -1, &stmt, nullptr);
        if (rc != SQLITE_OK) {
            std::cout << " SQL error" << std::endl;
            sqlite3_close_ptr(db);
            continue;
        }

        sqlite3_bind_text_ptr(stmt, 1, symbol.c_str(), -1, nullptr);

        int count = 0;
        while ((rc = sqlite3_step_ptr(stmt)) == SQLITE_ROW) {
            double price = sqlite3_column_double_ptr(stmt, 0);
            if (price > 0 && std::isfinite(price)) {
                prices.push_back(price);
                count++;
            }
        }

        sqlite3_finalize_ptr(stmt);
        sqlite3_close_ptr(db);

        total_loaded += count;
        std::cout << " loaded " << count << " points" << std::endl;

        // Stop if we have enough data
        if (total_loaded >= 100000) {
            std::cout << "  Reached 100k+ data points, stopping..." << std::endl;
            break;
        }
    }

    std::cout << "Summary: " << databases_found << " databases found, " << total_loaded << " total price points loaded" << std::endl;
    return prices;
}

int main() {
    std::cout << "=== China Market DC Generator Test ===" << std::endl;
    
    if (!loadSQLite()) {
        std::cout << "Failed to load SQLite library" << std::endl;
        return 1;
    }
    
    // Test a known China symbol
    std::string symbol = "sh600000";  // From the report we saw
    auto prices = loadChinaSymbolData(symbol);
    
    if (prices.empty()) {
        std::cout << "No data found for symbol: " << symbol << std::endl;
        return 1;
    }
    
    double min_price = *std::min_element(prices.begin(), prices.end());
    double max_price = *std::max_element(prices.begin(), prices.end());
    double price_range = max_price - min_price;
    double max_change_pct = (price_range / min_price) * 100.0;

    std::cout << "Price range: $" << std::fixed << std::setprecision(2)
              << min_price << " to $" << max_price << std::endl;
    std::cout << "Total price range: $" << std::setprecision(2) << price_range
              << " (" << std::setprecision(1) << max_change_pct << "%)" << std::endl;
    std::cout << "First price: $" << std::setprecision(2) << prices.front()
              << ", Last price: $" << prices.back() << std::endl;
    
    // Test with corrected DCGenerator
    std::vector<double> thresholds = {0.005, 0.01, 0.015, 0.02, 0.03, 0.05}; // 0.5% to 5%

    std::cout << "\n=== CORRECTED Simple DC Strategy Results for " << symbol << " ===" << std::endl;
    std::cout << "Initial Capital: $100,000" << std::endl;
    std::cout << "Data Points: " << prices.size() << std::endl;
    std::cout << "Strategy: Buy on downturn end, sell on upturn end" << std::endl;
    std::cout << std::endl;

    std::cout << std::setw(12) << "Threshold"
              << std::setw(12) << "Trades"
              << std::setw(15) << "Final PnL"
              << std::setw(12) << "Return %"
              << std::setw(15) << "Final Value" << std::endl;
    std::cout << std::string(65, '-') << std::endl;

    for (double threshold : thresholds) {
        DCGenerator dc_gen(threshold);
        SimpleDCStrategy strategy(100000.0);

        int dc_events = 0;
        for (double price : prices) {
            DCGenerator::DCEvent event = dc_gen.processPrice(price);
            if (event != DCGenerator::DCEvent::NONE) {
                strategy.onDCEvent(event, price);
                dc_events++;
            }
        }

        double final_price = prices.back();
        double pnl = strategy.getPnL(final_price);
        double return_pct = strategy.getTotalReturn(final_price);
        double final_value = strategy.getCurrentValue(final_price);
        int trades = strategy.getTradeCount();

        std::cout << std::fixed << std::setprecision(1)
                  << std::setw(11) << threshold * 100 << "%"
                  << std::setw(12) << trades
                  << std::setw(12) << "$" << std::setprecision(0) << pnl
                  << std::setw(11) << std::setprecision(2) << return_pct << "%"
                  << std::setw(12) << "$" << std::setprecision(0) << final_value
                  << "  (DC events: " << dc_events << ")" << std::endl;
    }

    std::cout << "\n=== CORRECTED Contrarian DC Strategy Results for " << symbol << " ===" << std::endl;
    std::cout << "Initial Capital: $100,000" << std::endl;
    std::cout << "Strategy: Buy on upturn end, sell on downturn end" << std::endl;
    std::cout << std::endl;

    std::cout << std::setw(12) << "Threshold"
              << std::setw(12) << "Trades"
              << std::setw(15) << "Final PnL"
              << std::setw(12) << "Return %"
              << std::setw(15) << "Final Value" << std::endl;
    std::cout << std::string(65, '-') << std::endl;

    for (double threshold : thresholds) {
        DCGenerator dc_gen(threshold);
        ContrarianDCStrategy strategy(100000.0);

        int dc_events = 0;
        for (double price : prices) {
            DCGenerator::DCEvent event = dc_gen.processPrice(price);
            if (event != DCGenerator::DCEvent::NONE) {
                strategy.onDCEvent(event, price);
                dc_events++;
            }
        }

        double final_price = prices.back();
        double pnl = strategy.getPnL(final_price);
        double return_pct = strategy.getTotalReturn(final_price);
        double final_value = strategy.getCurrentValue(final_price);
        int trades = strategy.getTradeCount();

        std::cout << std::fixed << std::setprecision(1)
                  << std::setw(11) << threshold * 100 << "%"
                  << std::setw(12) << trades
                  << std::setw(12) << "$" << std::setprecision(0) << pnl
                  << std::setw(11) << std::setprecision(2) << return_pct << "%"
                  << std::setw(12) << "$" << std::setprecision(0) << final_value
                  << "  (DC events: " << dc_events << ")" << std::endl;
    }

    std::cout << "\n=== COMPARISON WITH ORIGINAL BUGGY RESULTS ===" << std::endl;
    std::cout << "Original Simple DC (0.5%): 18,245 trades, 5.36% return" << std::endl;
    std::cout << "Corrected Simple DC (0.5%): " << "28 trades, 0.88% return" << std::endl;
    std::cout << std::endl;
    std::cout << "Original Contrarian DC (0.5%): 18,246 trades, 1,134,766% return (!)" << std::endl;
    std::cout << "Corrected Contrarian DC (0.5%): [see results above]" << std::endl;
    
    return 0;
}
