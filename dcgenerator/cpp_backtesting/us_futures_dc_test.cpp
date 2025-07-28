#include <iostream>
#include <vector>
#include <string>
#include <windows.h>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <set>

// SQLite function pointers
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

// Corrected DCGenerator
class DCGenerator {
public:
    enum class DCEvent {
        NONE,
        END_UPTURN,
        END_DOWNTURN
    };
    
    DCGenerator(double threshold) : threshold_(threshold), initialized_(false) {
        reset();
    }
    
    DCEvent processPrice(double price) {
        if (price <= 0.0) return DCEvent::NONE;
        
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
            }
        } else {
            if (current_price >= lowest_price_ * (1.0 + threshold_)) {
                return DCEvent::END_DOWNTURN;
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
        if (event == DCGenerator::DCEvent::END_DOWNTURN && current_cash_ > 0 && price > 0) {
            shares_held_ = current_cash_ / price;
            current_cash_ = 0;
            trade_count_++;
        } else if (event == DCGenerator::DCEvent::END_UPTURN && shares_held_ > 0 && price > 0) {
            current_cash_ = shares_held_ * price;
            shares_held_ = 0;
            trade_count_++;
        }
    }
    
    double getCurrentValue(double current_price) const {
        return current_cash_ + (shares_held_ * current_price);
    }
    
    double getTotalReturn(double current_price) const {
        return (getCurrentValue(current_price) - initial_capital_) / initial_capital_ * 100.0;
    }
    
    int getTradeCount() const { return trade_count_; }
};

// Contrarian DC Strategy
class ContrarianDCStrategy {
private:
    double initial_capital_;
    double current_cash_;
    double shares_held_;
    int trade_count_;
    
public:
    ContrarianDCStrategy(double capital) : initial_capital_(capital), current_cash_(capital), shares_held_(0), trade_count_(0) {}
    
    void onDCEvent(DCGenerator::DCEvent event, double price) {
        if (event == DCGenerator::DCEvent::END_UPTURN && current_cash_ > 0 && price > 0) {
            shares_held_ = current_cash_ / price;
            current_cash_ = 0;
            trade_count_++;
        } else if (event == DCGenerator::DCEvent::END_DOWNTURN && shares_held_ > 0 && price > 0) {
            current_cash_ = shares_held_ * price;
            shares_held_ = 0;
            trade_count_++;
        }
    }
    
    double getCurrentValue(double current_price) const {
        return current_cash_ + (shares_held_ * current_price);
    }
    
    double getTotalReturn(double current_price) const {
        return (getCurrentValue(current_price) - initial_capital_) / initial_capital_ * 100.0;
    }
    
    int getTradeCount() const { return trade_count_; }
};

// Explore US futures database structure
void exploreUSFuturesDatabase() {
    std::cout << "=== Exploring US Futures Database Structure ===" << std::endl;
    
    std::string db_path = "F:\\database\\us futures 1mins\\us_fut_1min.db";
    std::cout << "Database: " << db_path << std::endl;
    
    sqlite3* db;
    int rc = sqlite3_open_ptr(db_path.c_str(), &db);
    
    if (rc != SQLITE_OK) {
        std::cout << "Cannot open database: " << sqlite3_errmsg_ptr(db) << std::endl;
        if (db) sqlite3_close_ptr(db);
        return;
    }
    
    std::cout << "Database opened successfully!" << std::endl;
    
    // Get all tables
    const char* tables_sql = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name LIMIT 10;";
    sqlite3_stmt* stmt;
    
    rc = sqlite3_prepare_v2_ptr(db, tables_sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cout << "SQL error: " << sqlite3_errmsg_ptr(db) << std::endl;
        sqlite3_close_ptr(db);
        return;
    }
    
    std::cout << "\nFirst 10 tables found:" << std::endl;
    std::vector<std::string> tables;
    while ((rc = sqlite3_step_ptr(stmt)) == SQLITE_ROW) {
        const char* table_name = reinterpret_cast<const char*>(sqlite3_column_text_ptr(stmt, 0));
        if (table_name) {
            tables.push_back(std::string(table_name));
            std::cout << "  - " << table_name << std::endl;
        }
    }
    sqlite3_finalize_ptr(stmt);
    
    // Explore first table structure
    if (!tables.empty()) {
        std::string first_table = tables[0];
        std::cout << "\n=== Table: " << first_table << " ===" << std::endl;
        
        // Get column info
        std::string pragma_sql = "PRAGMA table_info(\"" + first_table + "\");";
        rc = sqlite3_prepare_v2_ptr(db, pragma_sql.c_str(), -1, &stmt, nullptr);
        
        if (rc == SQLITE_OK) {
            std::cout << "Columns:" << std::endl;
            while ((rc = sqlite3_step_ptr(stmt)) == SQLITE_ROW) {
                const char* col_name = reinterpret_cast<const char*>(sqlite3_column_text_ptr(stmt, 1));
                const char* col_type = reinterpret_cast<const char*>(sqlite3_column_text_ptr(stmt, 2));
                if (col_name && col_type) {
                    std::cout << "  " << col_name << " (" << col_type << ")" << std::endl;
                }
            }
            sqlite3_finalize_ptr(stmt);
        }
        
        // Get sample data
        std::string sample_sql = "SELECT * FROM \"" + first_table + "\" LIMIT 5;";
        rc = sqlite3_prepare_v2_ptr(db, sample_sql.c_str(), -1, &stmt, nullptr);
        
        if (rc == SQLITE_OK) {
            std::cout << "Sample data:" << std::endl;
            int row_count = 0;
            while ((rc = sqlite3_step_ptr(stmt)) == SQLITE_ROW && row_count < 5) {
                std::cout << "  Row " << (row_count + 1) << ": ";
                int col_count = 6; // Assume typical OHLCV + symbol + time structure
                for (int col = 0; col < col_count; ++col) {
                    const char* value = reinterpret_cast<const char*>(sqlite3_column_text_ptr(stmt, col));
                    std::cout << (value ? value : "NULL") << " | ";
                }
                std::cout << std::endl;
                row_count++;
            }
            sqlite3_finalize_ptr(stmt);
        }
        
        // Get unique symbols from futures_data table
        std::string symbols_sql = "SELECT DISTINCT symbol FROM futures_data ORDER BY symbol LIMIT 20;";
        rc = sqlite3_prepare_v2_ptr(db, symbols_sql.c_str(), -1, &stmt, nullptr);

        if (rc == SQLITE_OK) {
            std::cout << "Available futures symbols:" << std::endl;
            while ((rc = sqlite3_step_ptr(stmt)) == SQLITE_ROW) {
                const char* symbol = reinterpret_cast<const char*>(sqlite3_column_text_ptr(stmt, 0));
                if (symbol) {
                    std::cout << "  " << symbol << std::endl;
                }
            }
            sqlite3_finalize_ptr(stmt);
        }

        // Get total record count
        std::string count_sql = "SELECT COUNT(*) FROM futures_data;";
        rc = sqlite3_prepare_v2_ptr(db, count_sql.c_str(), -1, &stmt, nullptr);

        if (rc == SQLITE_OK && sqlite3_step_ptr(stmt) == SQLITE_ROW) {
            const char* count_str = reinterpret_cast<const char*>(sqlite3_column_text_ptr(stmt, 0));
            if (count_str) {
                std::cout << "Total futures records: " << count_str << std::endl;
            }
            sqlite3_finalize_ptr(stmt);
        }
    }
    
    sqlite3_close_ptr(db);
}

// Load US futures data for a specific symbol
std::vector<double> loadUSFuturesData(const std::string& symbol) {
    std::vector<double> prices;

    std::string db_path = "F:\\database\\us futures 1mins\\us_fut_1min.db";
    std::cout << "Loading futures data for: " << symbol << std::endl;

    sqlite3* db;
    int rc = sqlite3_open_ptr(db_path.c_str(), &db);

    if (rc != SQLITE_OK) {
        std::cout << "Cannot open futures database: " << sqlite3_errmsg_ptr(db) << std::endl;
        if (db) sqlite3_close_ptr(db);
        return prices;
    }

    // Query the single futures_data table
    std::string data_sql = "SELECT close FROM futures_data WHERE symbol = ? ORDER BY datetime LIMIT 100000;";
    sqlite3_stmt* stmt;

    rc = sqlite3_prepare_v2_ptr(db, data_sql.c_str(), -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cout << "Failed to prepare SQL: " << sqlite3_errmsg_ptr(db) << std::endl;
        sqlite3_close_ptr(db);
        return prices;
    }

    sqlite3_bind_text_ptr(stmt, 1, symbol.c_str(), -1, nullptr);

    std::cout << "Querying futures_data table..." << std::endl;

    int count = 0;
    while ((rc = sqlite3_step_ptr(stmt)) == SQLITE_ROW) {
        double price = sqlite3_column_double_ptr(stmt, 0);
        if (price > 0 && std::isfinite(price)) {
            prices.push_back(price);
            count++;

            // Progress update every 10,000 points
            if (count % 10000 == 0) {
                std::cout << "  Loaded " << count << " points..." << std::endl;
            }
        }
    }
    sqlite3_finalize_ptr(stmt);
    sqlite3_close_ptr(db);

    std::cout << "Total loaded: " << prices.size() << " price points" << std::endl;
    return prices;
}

// Test a specific futures symbol
void testFuturesSymbol(const std::string& symbol) {
    std::cout << "\n=== Testing Futures Symbol: " << symbol << " ===" << std::endl;

    auto prices = loadUSFuturesData(symbol);

    if (prices.size() < 1000) {
        std::cout << "Insufficient data (" << prices.size() << " points)" << std::endl;
        return;
    }

    double min_price = *std::min_element(prices.begin(), prices.end());
    double max_price = *std::max_element(prices.begin(), prices.end());
    double price_range_pct = ((max_price - min_price) / min_price) * 100.0;

    std::cout << "Data summary:" << std::endl;
    std::cout << "  Price range: $" << std::fixed << std::setprecision(2) << min_price << " to $" << max_price << std::endl;
    std::cout << "  Price range %: " << std::setprecision(1) << price_range_pct << "%" << std::endl;
    std::cout << "  Data points: " << prices.size() << std::endl;

    // Test with futures-appropriate thresholds
    std::vector<double> thresholds = {0.005, 0.01, 0.015, 0.02, 0.03, 0.05}; // 0.5% to 5%

    std::cout << "\n=== Simple DC Strategy Results ===" << std::endl;
    std::cout << "Initial Capital: $100,000" << std::endl;
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
        double return_pct = strategy.getTotalReturn(final_price);
        double final_value = strategy.getCurrentValue(final_price);
        double pnl = final_value - 100000.0;
        int trades = strategy.getTradeCount();

        std::cout << std::fixed << std::setprecision(1)
                  << std::setw(11) << threshold * 100 << "%"
                  << std::setw(12) << trades
                  << std::setw(12) << "$" << std::setprecision(0) << pnl
                  << std::setw(11) << std::setprecision(2) << return_pct << "%"
                  << std::setw(12) << "$" << std::setprecision(0) << final_value
                  << "  (DC events: " << dc_events << ")" << std::endl;
    }

    std::cout << "\n=== Contrarian DC Strategy Results ===" << std::endl;
    std::cout << "Initial Capital: $100,000" << std::endl;
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
        double return_pct = strategy.getTotalReturn(final_price);
        double final_value = strategy.getCurrentValue(final_price);
        double pnl = final_value - 100000.0;
        int trades = strategy.getTradeCount();

        std::cout << std::fixed << std::setprecision(1)
                  << std::setw(11) << threshold * 100 << "%"
                  << std::setw(12) << trades
                  << std::setw(12) << "$" << std::setprecision(0) << pnl
                  << std::setw(11) << std::setprecision(2) << return_pct << "%"
                  << std::setw(12) << "$" << std::setprecision(0) << final_value
                  << "  (DC events: " << dc_events << ")" << std::endl;
    }
}

int main() {
    std::cout << "=== US Futures 1-Minute DC Testing ===" << std::endl;

    if (!loadSQLite()) {
        std::cout << "Failed to load SQLite library" << std::endl;
        return 1;
    }

    // First explore the database structure
    exploreUSFuturesDatabase();

    std::cout << "\n" << std::string(60, '=') << std::endl;

    // Test some common futures symbols
    std::vector<std::string> test_symbols = {
        "ES",    // S&P 500 E-mini
        "NQ",    // NASDAQ 100 E-mini
        "YM",    // Dow Jones E-mini
        "CL",    // Crude Oil
        "GC",    // Gold
        "SI",    // Silver
        "ZN",    // 10-Year Treasury Note
        "ZB",    // 30-Year Treasury Bond
        "6E",    // Euro
        "6J"     // Japanese Yen
    };

    std::cout << "\nTesting common futures symbols..." << std::endl;
    std::cout << "Note: Will test first available symbol from the list" << std::endl;

    // Test the first available symbol
    for (const auto& symbol : test_symbols) {
        auto prices = loadUSFuturesData(symbol);
        if (prices.size() >= 1000) {
            testFuturesSymbol(symbol);
            break;
        } else {
            std::cout << "Symbol " << symbol << " has insufficient data (" << prices.size() << " points)" << std::endl;
        }
    }

    std::cout << "\n=== US Futures DC Testing Completed ===" << std::endl;
    std::cout << "Results show corrected DCGenerator performance on futures data" << std::endl;

    return 0;
}
