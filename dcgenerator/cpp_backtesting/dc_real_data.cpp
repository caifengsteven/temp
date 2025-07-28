#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <algorithm>
#include "C:/sqlite3/sqlite3.h"

// Simple DC Generator
class DCGenerator {
public:
    enum class DCEvent { NONE, END_UPTURN, END_DOWNTURN };
    
    explicit DCGenerator(double threshold) : threshold_(threshold), initialized_(false) {}
    
    DCEvent processPrice(double price) {
        if (!std::isfinite(price) || price <= 0) return DCEvent::NONE;
        
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
    
    void reset() {
        initialized_ = false;
        highest_price_ = 0;
        lowest_price_ = 0;
        is_upturn_ = true;
    }

private:
    double threshold_;
    bool initialized_;
    double highest_price_;
    double lowest_price_;
    bool is_upturn_;
};

// Trading data structure
struct TradeData {
    std::string date;
    std::string time;
    double price;
    int volume;
    std::string buysell;
};

// Simple trading strategy
class TradingStrategy {
public:
    explicit TradingStrategy(double initial_capital) 
        : initial_capital_(initial_capital), cash_(initial_capital), position_(0), trade_count_(0) {}
    
    void onDCEvent(DCGenerator::DCEvent event, double price, const std::string& date, const std::string& time) {
        if (!std::isfinite(price) || price <= 0) return;
        
        switch (event) {
            case DCGenerator::DCEvent::END_DOWNTURN:
                if (position_ == 0 && cash_ > 0) {
                    double cash_to_use = cash_ * 0.95;
                    position_ = cash_to_use / price;
                    cash_ -= cash_to_use;
                    trade_count_++;
                    std::cout << "BUY at $" << std::fixed << std::setprecision(2) << price 
                              << " on " << date << " " << time 
                              << " (Position: " << std::setprecision(4) << position_ << ")" << std::endl;
                }
                break;
                
            case DCGenerator::DCEvent::END_UPTURN:
                if (position_ > 0) {
                    cash_ += position_ * price;
                    trade_count_++;
                    std::cout << "SELL at $" << std::fixed << std::setprecision(2) << price 
                              << " on " << date << " " << time 
                              << " (Cash: $" << std::setprecision(2) << cash_ << ")" << std::endl;
                    position_ = 0;
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
        return (getCurrentValue(current_price) - initial_capital_) / initial_capital_ * 100.0;
    }
    
    int getTradeCount() const { return trade_count_; }

private:
    double initial_capital_;
    double cash_;
    double position_;
    int trade_count_;
};

// Database functions
std::vector<std::string> getAvailableSymbols(sqlite3* db) {
    std::vector<std::string> symbols;
    
    const char* sql = "SELECT DISTINCT symbol FROM trade ORDER BY symbol LIMIT 20;";
    sqlite3_stmt* stmt;
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "SQL error: " << sqlite3_errmsg(db) << std::endl;
        return symbols;
    }
    
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        const char* symbol = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        if (symbol) {
            symbols.push_back(std::string(symbol));
        }
    }
    
    sqlite3_finalize(stmt);
    return symbols;
}

std::vector<TradeData> loadTradingData(sqlite3* db, const std::string& symbol, int limit = 50000) {
    std::vector<TradeData> trades;
    
    const char* sql = "SELECT date, time, price, volume, buysell FROM trade WHERE symbol = ? ORDER BY date, time LIMIT ?;";
    sqlite3_stmt* stmt;
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "SQL error: " << sqlite3_errmsg(db) << std::endl;
        return trades;
    }
    
    sqlite3_bind_text(stmt, 1, symbol.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 2, limit);
    
    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        TradeData trade;
        
        const char* date = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        const char* time = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        const char* buysell = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        
        trade.date = date ? std::string(date) : "";
        trade.time = time ? std::string(time) : "";
        trade.price = sqlite3_column_double(stmt, 2);
        trade.volume = sqlite3_column_int(stmt, 3);
        trade.buysell = buysell ? std::string(buysell) : "";
        
        trades.push_back(trade);
    }
    
    sqlite3_finalize(stmt);
    return trades;
}

void testDCWithRealData(const std::vector<TradeData>& trades, const std::string& symbol, double threshold) {
    std::cout << "\n=== Testing DC Algorithm with Real Data ===" << std::endl;
    std::cout << "Symbol: " << symbol << std::endl;
    std::cout << "Threshold: " << threshold * 100 << "%" << std::endl;
    std::cout << "Data points: " << trades.size() << std::endl;
    
    if (trades.empty()) {
        std::cout << "No data to process!" << std::endl;
        return;
    }
    
    // Find price range
    double min_price = trades[0].price;
    double max_price = trades[0].price;
    for (const auto& trade : trades) {
        min_price = std::min(min_price, trade.price);
        max_price = std::max(max_price, trade.price);
    }
    
    std::cout << "Price range: $" << std::fixed << std::setprecision(2) 
              << min_price << " to $" << max_price << std::endl;
    std::cout << "Date range: " << trades[0].date << " to " << trades.back().date << std::endl;
    
    // Process with DC algorithm
    DCGenerator dc_gen(threshold);
    TradingStrategy strategy(100000.0);
    
    int dc_events = 0;
    int upturn_ends = 0;
    int downturn_ends = 0;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < trades.size(); ++i) {
        DCGenerator::DCEvent event = dc_gen.processPrice(trades[i].price);
        
        if (event != DCGenerator::DCEvent::NONE) {
            dc_events++;
            if (event == DCGenerator::DCEvent::END_UPTURN) {
                upturn_ends++;
            } else if (event == DCGenerator::DCEvent::END_DOWNTURN) {
                downturn_ends++;
            }
            
            strategy.onDCEvent(event, trades[i].price, trades[i].date, trades[i].time);
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Processing time: " << duration.count() << " microseconds" << std::endl;
    std::cout << "Total DC events: " << dc_events << std::endl;
    std::cout << "End upturn events: " << upturn_ends << std::endl;
    std::cout << "End downturn events: " << downturn_ends << std::endl;
    std::cout << "Events per 1000 ticks: " << (dc_events * 1000.0 / trades.size()) << std::endl;
    
    // Trading results
    double final_return = strategy.getTotalReturn(trades.back().price);
    std::cout << "\n=== Trading Strategy Results ===" << std::endl;
    std::cout << "Total return: " << std::fixed << std::setprecision(2) << final_return << "%" << std::endl;
    std::cout << "Number of trades: " << strategy.getTradeCount() << std::endl;
}

int main() {
    std::cout << "=== DC Generator Real Database Test ===" << std::endl;
    
    const std::string db_path = "I:/zhubi/cpp_implementation/sqlite_databases/2018/2018_01.db";
    
    // Open database
    sqlite3* db;
    int rc = sqlite3_open(db_path.c_str(), &db);
    
    if (rc != SQLITE_OK) {
        std::cerr << "Cannot open database: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        return 1;
    }
    
    std::cout << "✅ Connected to database: " << db_path << std::endl;
    
    try {
        // Get available symbols
        std::cout << "\n=== Available Symbols ===" << std::endl;
        auto symbols = getAvailableSymbols(db);
        
        if (symbols.empty()) {
            std::cout << "No symbols found in database!" << std::endl;
            sqlite3_close(db);
            return 1;
        }
        
        std::cout << "Found " << symbols.size() << " symbols:" << std::endl;
        for (size_t i = 0; i < std::min(symbols.size(), size_t(10)); ++i) {
            std::cout << "  " << (i+1) << ". " << symbols[i] << std::endl;
        }
        
        // Use the first symbol for testing
        std::string test_symbol = symbols[0];
        std::cout << "\n=== Loading data for: " << test_symbol << " ===" << std::endl;
        
        auto trades = loadTradingData(db, test_symbol, 100000);
        
        if (trades.empty()) {
            std::cout << "No trading data found for " << test_symbol << std::endl;
        } else {
            std::cout << "✅ Loaded " << trades.size() << " trades" << std::endl;
            
            // Test with different thresholds
            std::vector<double> thresholds = {0.001, 0.002, 0.005};
            
            for (double threshold : thresholds) {
                testDCWithRealData(trades, test_symbol, threshold);
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    
    sqlite3_close(db);
    std::cout << "\n✅ Database connection closed." << std::endl;
    
    return 0;
}
